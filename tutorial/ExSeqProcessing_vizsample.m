
TRANSCRIPT_OBJ_FILE = ‘’; %put in the _transcriptobjects.mat output in the 6_base-calling directory
XY_SIZE = [2048,2048]; %How big (by pixels) is a single image from microscope
READ_SIZE = 8; %how big dots do you want to make?

%Load transcript_objects into the memory:
load(TRANSCRIPT_OBJ_FILE);

%Get the list of all unique gene names in this sample 
insitu_genes = cellfun(@(x) x.name, transcript_objects,'UniformOutput',0);
insitu_genes_unique = unique(insitu_genes);

%To make circle markers instead of squares, we add this code
t = strel('disk',READ_SIZE+1);
circleShape = t.Neighborhood;

    

outdir = fullfile('gene_maps/');
if ~exist(outdir,'dir')
   mkdir(outdir)
end
    
    
for g_idx = 1:length(insitu_genes_unique)
    read_count = 0;
    geneMap = zeros(XY_SIZE);
        
    for r_idx = 1:length(transcript_objects)

        if strcmp(transcript_objects{r_idx}.name,insitu_genes_unique{g_idx})
            gpos = round(transcript_objects{r_idx}.pos);
            gpos_max = gpos+READ_SIZE;
            gpos_min = gpos-READ_SIZE;
                
            %make sure dots don't go outside the image
            gpos_min(1) = max(1,gpos_min(1));
            gpos_min(2) = max(1,gpos_min(2));
            gpos_max(1) = min(size(geneMap,1),gpos_max(1));
            gpos_max(2) = min(size(geneMap,2),gpos_max(2));
                
            %Add overlapping reads
            try
                geneMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2),1) = ...
                    geneMap(gpos_min(1):gpos_max(1),gpos_min(2):gpos_max(2),1) + ...
                    circleShape;
            end
            read_count = read_count+1;
        end
   end %end loop over transcript_objcets
%         geneMap_ds = imresize3(geneMap,1/2.,'nearest');
        save3DTif_uint16(geneMap, fullfile(outdir, sprintf('genemap_%s.tif',insitu_genes_unique{g_idx})));
        fprintf('Created map for gene %s with %i reads\n',insitu_genes_unique{g_idx},read_count);
end %end loop over genes in thes sample

% To recreate figures from the paper, you can load these genemaps in FIJI, then assign each gene a distinct color. You do this using Image > Color > Merge Colors.  
