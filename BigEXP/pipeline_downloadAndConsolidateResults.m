%Load the 

%% Download all the data
%uses yamlfile,save_dir, called as a script to help debug

combined_transcriptsfile_refined = strrep(combined_transcriptsfile,'.mat','-h1.mat');

if ~exist(combined_transcriptsfile,'file') && ~exist(combined_transcriptsfile_refined,'file')
downloadAllTranscripts; 
end
%% Limit the reads to only hamming<=1

if ~exist(combined_transcriptsfile_refined, 'file')
    load(combined_transcriptsfile); %load transcript_objects_light
    h1hits = cell2mat(cellfun(@(x) x.hamming<=1, transcript_objects_light,'UniformOutput',0));
    transcript_objects_light = transcript_objects_light(h1hits);


    save(combined_transcriptsfile_refined,'transcript_objects_light','-v7.3');
    fprintf('%i of %i transcript objects are kept\n',length(transcript_objects_light),length(h1hits));

    delete(combined_transcriptsfile); %Save space by deleting the non-H1 reads

else
    load(combined_transcriptsfile_refined);
end

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
    
%     try
        geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2)) = ...
            geneHeatMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2))+10;
%     end
end

heatmap_filepath = fullfile(OUTDIR, sprintf('htapp%i_genemap_%s.tif',EXP_NUM,'TOTALSUM'));
save3DTif_uint16(geneHeatMap,heatmap_filepath);
fprintf('Created map for all %i reads\n',length(transcript_objects_light));

%%
img_ds_dapi = load3DImage_uint16(imgfile_ds_dapi);
rgb_debug = zeros(size(img_ds_dapi,1),size(img_ds_dapi,2),3);
rgb_debug(:,:,1) = min(img_ds_dapi./200,1.);
rgb_debug(:,:,2) = min(geneHeatMap./10,1.);
figure
imshow(rgb_debug)
%% Apply the manual seg
[transcript_objects_2Dseg,file_path_segmentedtranscriptobjects] = exseq_applyManualNuclearSeg(DOWNSAMPLE_RATE,imgfile_ds_dapi,imgfile_ds_seg,transcript_objects_light,OUTDIR);
fprintf('Manual segmented file location: %s\n',file_path_segmentedtranscriptobjects);
%% 
exseq_exportDataToR(save_dir,save_dir,file_path_segmentedtranscriptobjects)

%% View the highest genes in the cell seg
if ~exist('transcript_objects_2Dseg','var')
    load(file_path_segmentedtranscriptobjects);
end

insitu_genes = cellfun(@(x) x.name, transcript_objects_2Dseg,'UniformOutput',0);
insitu_genes_cat = categorical(insitu_genes);
figure; 

h = histogram(insitu_genes_cat,'DisplayOrder','descend');
title(sprintf('%i in situ reads in segmented cells of %s experiment',length(insitu_genes_cat), experiment_name));

min_count_to_see = 1000; %only visualize the genes with 1000 reads or more
genes_seen = {};
genes_count = [];
for g_idx = 1:length(h.Values)
    if h.Values(g_idx)<min_count_to_see
        break
    end
    fprintf('%s\t%i\n',h.Categories{g_idx},h.Values(g_idx));
    genes_seen{g_idx} = h.Categories{g_idx};
    genes_count(g_idx) = h.Values(g_idx);
end

fprintf('\nTop expressed genes to load into Seurat\n');

num_genes_to_see = g_idx-1;
%Make a gene list for loading into Seurat
for g_idx = 1:num_genes_to_see
fprintf('''%s'',',genes_seen{g_idx});
end

%% Write a different file that writes gene, xyz position, cell_id
filename = fullfile(OUTDIR,sprintf('htapp%i_geneSpaceCell.csv',EXP_NUM));
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