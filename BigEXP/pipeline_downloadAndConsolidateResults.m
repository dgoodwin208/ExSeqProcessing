%Load the

%% Download all the data
%uses yamlfile,save_dir, called as a script to help debug

combined_transcriptsfile_refined = strrep(combined_transcriptsfile,'.mat',sprintf('-h%i.mat',USABLE_HAMMING));

if ~exist(combined_transcriptsfile,'file') && ~exist(combined_transcriptsfile_refined,'file')
    transcript_objects_light = downloadAllTranscripts(yamlfile,save_dir,combined_transcriptsfile);
end
%% Limit the reads to only hamming<=USABLE_HAMMING

if ~exist(combined_transcriptsfile_refined, 'file')
    load(combined_transcriptsfile); %load transcript_objects_light
    h1hits = cell2mat(cellfun(@(x) x.hamming<=USABLE_HAMMING, transcript_objects_light,'UniformOutput',0));
    transcript_objects_light = transcript_objects_light(h1hits);
    
    
    save(combined_transcriptsfile_refined,'transcript_objects_light','-v7.3');
    fprintf('%i of %i transcript objects are kept\n',length(transcript_objects_light),length(h1hits));
    
    %delete(combined_transcriptsfile); %Save space by deleting the non-H1 reads
    
else
    load(combined_transcriptsfile_refined);
end
%% Remove global positioned reads that have an erroneous global position, 
%specifically a low Z value
pos_flag = zeros(size(transcript_objects_light));
for t_idx = 1:length(transcript_objects_light)
    tobj = transcript_objects_light{t_idx};

    %downsample the global position so we can identify cell_id
    pos = round(tobj.globalpos/DOWNSAMPLE_RATE);
    %Note that some read positions if they are less than ~5 in XYZ pixel
    %value, will be wiped, but that's probably fine to remove those anyway
    %in case of weird artifacts
    if any(pos==0)
        fprintf('Discarding read at downsampled pos: %s\n',mat2str(pos));
        pos_flag(t_idx) = 1;
    end
end
fprintf('Removed %i/%i reads for having a strange position\n',sum(pos_flag),length(transcript_objects_light));
transcript_objects_light(logical(pos_flag)) = [];

%% Get all the gene names so we can make a (sorted!) histogram

insitu_genes = cellfun(@(x) x.name, transcript_objects_light,'UniformOutput',0);
insitu_genes_cat = categorical(insitu_genes);
figure;

h = histogram(insitu_genes_cat,'DisplayOrder','descend');
title(sprintf('%i in situ reads in %s experiment',length(insitu_genes_cat), experiment_name));

genes_seen = cell(length(h.Values),1);
genes_count = zeros(length(h.Values),1);
for g_idx = 1:length(h.Values)
    fprintf('%s\t%i\n',h.Categories{g_idx},h.Values(g_idx));
    genes_seen{g_idx} = h.Categories{g_idx};
    genes_count(g_idx) = h.Values(g_idx);
end

%% Write a CSV that writes gene, xyz position for analysis outside of MATLAB
filename = fullfile(ROOTDIR,sprintf('%s_geneSpace.csv',experiment_name));
fid = fopen(filename,'w');
for t_idx = 1:length(transcript_objects_light)
    tobj = transcript_objects_light{t_idx};
    tobj.globalpos = round( tobj.globalpos);
    fprintf(fid,'%s,%i,%i,%i\n',...
        tobj.name,...
        tobj.globalpos(1),tobj.globalpos(2),tobj.globalpos(3));
end
fclose(fid);
fprintf('Saved CSV file: %s\n',filename);

%% Make a map of all the filtered, summing all reads to get a heatmap
READ_SIZE = 1;
imdim = image_dimensions(imgfile_ds_dapi); imdim = imdim([2,1,3]);
geneHeatMap = zeros(imdim);
for r_idx = 1:length(transcript_objects_light)
    gpos = round(transcript_objects_light{r_idx}.globalpos./DOWNSAMPLE_RATE);
    gpos_max = gpos+READ_SIZE;
    gpos_min = gpos-READ_SIZE;
    
    %clamp the values
    gpos_min(1) = max(1,gpos_min(1));
    gpos_min(2) = max(1,gpos_min(2));
    gpos_max(1) = min(size(geneHeatMap,1),gpos_max(1));
    gpos_max(2) = min(size(geneHeatMap,2),gpos_max(2));
    
    geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2)) = ...
        geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2))+10;
    
end

heatmap_filepath = fullfile(ROOTDIR, sprintf('%s_genemap_%s.tif',experiment_name,'TOTALSUM'));
save3DTif_uint16(geneHeatMap,heatmap_filepath);
fprintf('Created map for all %i reads\n',length(transcript_objects_light));
geneHeatMap_total = geneHeatMap;
%%
READ_SIZE = 6;
% DOWNSAMPLE_RATE = 16;
outdir = fullfile(save_dir,'downsample_maps');
if ~exist(outdir)
    mkdir(outdir)
end

insitu_genes_unique = unique(insitu_genes);

for g_idx = 1:length(insitu_genes_unique)
    read_count = 0;
    geneHeatMap = zeros(imdim);
    for r_idx = 1:length(transcript_objects_light)
        if strcmp(transcript_objects_light{r_idx}.name,insitu_genes_unique{g_idx})
            gpos = round(transcript_objects_light{r_idx}.globalpos./DOWNSAMPLE_RATE);
            gpos_max = gpos+READ_SIZE;
            gpos_min = gpos-READ_SIZE;
            
            %clamp the values
            gpos_min(1) = max(1,gpos_min(1));
            gpos_min(2) = max(1,gpos_min(2));
            gpos_max(1) = min(size(geneHeatMap,1),gpos_max(1));
            gpos_max(2) = min(size(geneHeatMap,2),gpos_max(2));
            
            geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2)) = ...
                geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2))+10;

            read_count = read_count+1;
        end
    end
    geneMap_ds = imresize3(geneHeatMap,1/DOWNSAMPLE_RATE,'nearest');
    save3DTif_uint16(geneMap_ds, fullfile(outdir, sprintf('genemap_%s.tif',insitu_genes_unique{g_idx})));
    fprintf('Created map for gene %s with %i reads\n',insitu_genes_unique{g_idx},read_count);
end


%%
img_ds_dapi = load3DImage_uint16(imgfile_ds_dapi);
rgb_debug = zeros(size(img_ds_dapi,1),size(img_ds_dapi,2),3);
rgb_debug(:,:,1) = min(img_ds_dapi./200,1.);
rgb_debug(:,:,2) = min(geneHeatMap_total./10,1.);
figure
imshow(rgb_debug)
%% Apply the manual seg

[transcript_objects_cellids,cell_centroids,imgSeg_filtered] = exseq_applyManualNuclearSeg(imgfile_ds_dapi,imgfile_ds_seg,transcript_objects_light);

cellmap_filename = strrep(imgfile_ds_seg,'.tif','_NUMBERED.tif');
save3DImage_uint16(imgSeg_filtered,cellmap_filename);
% Get just the segmented cells
transcript_objects_2Dseg = transcript_objects_cellids;
%transcript_objects to remove
tobjs_indices_to_remove = find(cell2mat(cellfun(@(x) [x.cell_id==0],transcript_objects_2Dseg,'UniformOutput',false)));
fprintf('Removed %i/%i entries that were not in an annotated cell \n', length(tobjs_indices_to_remove), length(transcript_objects_cellids));
transcript_objects_2Dseg(tobjs_indices_to_remove) = [];


save(segged_outfile,'transcript_objects_2Dseg');
fprintf('Finished saving the transcripts\n');

fprintf('Done segmenting: %i reads in %i cells: %s \n',length(transcript_objects_2Dseg),size(cell_centroids,1));

%% View the cell segmented reads in space

insitu_genes_cellseg = cellfun(@(x) x.name, transcript_objects_2Dseg,'UniformOutput',0);

READ_SIZE = 6;
outdir = fullfile(save_dir,'downsample_maps_segmented');
if ~exist(outdir)
    mkdir(outdir)
end

insitu_genes_cellseg_unique = unique(insitu_genes_cellseg);

for g_idx = 1:length(insitu_genes_cellseg_unique)
    read_count = 0;
    geneHeatMap = zeros(imdim);
    for r_idx = 1:length(transcript_objects_2Dseg)
        if strcmp(transcript_objects_light{r_idx}.name,insitu_genes_cellseg_unique{g_idx})
            gpos = round(transcript_objects_2Dseg{r_idx}.globalpos./DOWNSAMPLE_RATE);
            gpos_max = gpos+READ_SIZE;
            gpos_min = gpos-READ_SIZE;
            
            %clamp the values
            gpos_min(1) = max(1,gpos_min(1));
            gpos_min(2) = max(1,gpos_min(2));
            gpos_max(1) = min(size(geneHeatMap,1),gpos_max(1));
            gpos_max(2) = min(size(geneHeatMap,2),gpos_max(2));
            
            geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2)) = ...
                geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2))+10;

            read_count = read_count+1;
        end
    end
    geneMap_ds = imresize3(geneHeatMap,1/DOWNSAMPLE_RATE,'nearest');
    save3DTif_uint16(geneMap_ds, fullfile(outdir, sprintf('genemap_%s.tif',insitu_genes_cellseg_unique{g_idx})));
    fprintf('Created map for gene %s with %i reads\n',insitu_genes_cellseg_unique{g_idx},read_count);
end


%%
exseq_exportDataToR(transcript_objects_2Dseg,ROOTDIR)

%% View the highest genes in the cell seg

if ~exist('transcript_objects_2Dseg','var')
    load(file_path_segmentedtranscriptobjects);
end

insitu_genes = cellfun(@(x) x.name, transcript_objects_2Dseg,'UniformOutput',0);
insitu_genes_cat = categorical(insitu_genes);
figure;

h = histogram(insitu_genes_cat,'DisplayOrder','descend');
title(sprintf('%i in situ reads in segmented cells of %s experiment',length(insitu_genes_cat), experiment_name));

min_count_to_see = 0; %only visualize the genes with 1000 reads or more
genes_seen = {};
genes_count = [];
fid = fopen('gene_count.csv','w');
for g_idx = 1:length(h.Values)
    if h.Values(g_idx)<min_count_to_see
        break
    end
    fprintf('%s\t%i\n',h.Categories{g_idx},h.Values(g_idx));
    genes_seen{g_idx} = h.Categories{g_idx};
    genes_count(g_idx) = h.Values(g_idx);
    fprintf(fid,'%s,%i\n',h.Categories{g_idx},h.Values(g_idx));
end
fclose(fid);

% fprintf('\nTop expressed genes to load into Seurat\n');
% num_genes_to_see = g_idx-1;
% %Make a gene list for loading into Seurat
% for g_idx = 1:num_genes_to_see
%     fprintf('''%s'',',genes_seen{g_idx});
% end

%% Write a different file that writes gene, xyz position, cell_id
filename = fullfile(ROOTDIR,sprintf('%s_geneSpaceCell.csv',experiment_name));
fid = fopen(filename,'w');
for t_idx = 1:length(transcript_objects_2Dseg)
    tobj = transcript_objects_2Dseg{t_idx};
    tobj.globalpos = round( tobj.globalpos);
    fprintf(fid,'%s,%i,%i,%i,%i\n',...
        tobj.name,...
        tobj.globalpos(1),tobj.globalpos(2),tobj.globalpos(3),...
        tobj.cell_id);
end
fclose(fid);