%% Load the transcript objects

loadParameters;

%The options are: 'splintr3F0crop',splintr3F1crop','splintr3F2'
params.FILE_BASENAME = 'splintr3F0crop';

RESULTSDIR = sprintf('/Users/goody/Neuro/ExSeq/splintr/%s/',params.FILE_BASENAME);
%Load the set of all pixels of all puncta across all rounds from the
filename_pixels = sprintf(fullfile(RESULTSDIR,sprintf('%s_punctavoxels.mat',params.FILE_BASENAME)));

load(filename_pixels)

fprintf('Loaded.\n');

%This is the size of the cropped images, corresponding to ~15um pre-ExM
IMG_SIZE = [2048 2048 100];

%We're reading 4-base barcodes
readlength =4;

% The _punctavoxels.mat file contains cell arrays that include the voxel
% locations (stored as 1D indices) and the color values (called puncta_set)
% Here we initialize the _filtered arrays, which will be modified later
puncta_indices_cell_filtered = puncta_indices_cell;
puncta_set_cell_filtered = puncta_set_cell;

%Can we utilize the fact that we know Red (chan1) can be high without
%Magenta (chan 2) being high, but Red goes high whenever magenta goes high? 
%-1 means no correction, .3 means switch from 1->2 if 2 within 30% of 1's 
%brightness. (This is used in the normalization_qnorm.m script)
ILLUMINACORRECTIONFACTOR = -1;


%Keep track of funnel numbers:
% [original number of puncta, 
% number removed missing bases, 
% number aligned,
% shuffled_aligned]
funnel_numbers = zeros(4,1);
funnel_names = {'Segmented amplicons','Present in every round',...
    'Aligned to Barcodes','Column shuffled hits'};

%% Load the 6-mer barcodes, which we then need to shrink to just 4-mer
load('groundtruth_dictionary_splintr20180621.mat')

[unique_codes,ia,ic] = unique(groundtruth_codes(:,1:4),'rows');

num_unique_barcodes = max(ic);
gtlabels_4mers = cell(1,num_unique_barcodes);

%Sort the unique
[ic_sorted, indices] = sort(ic,'ascend');

for unique_code_index = 1:num_unique_barcodes
    %Get the first sorted index for this barcode number
    start_idx = find(ic_sorted==unique_code_index,1);
    %Get the original string, then crop the padlock id from it
    string_total = gtlabels{indices(start_idx)};
    string_parts = split(string_total,'_');
    gtlabels_4mers{ic_sorted(start_idx)} = string_parts{1};
end

%When the barcodes were designed, the bases were arbitrary and map to the
%microscope channels in the following order
barcode_to_microscope_mapping = [2, 1, 4, 3];

% Looks good as compared to Oz's Barcodes_01 file!
% overwrite the 6mer variables
% groundtruth_codes = unique_codes;
groundtruth_codes = barcode_to_microscope_mapping(unique_codes);


%Print out the results for comparison (shifting back to 0s for ease)
for unique_code_index = 1:num_unique_barcodes
    fprintf('%s\t%s\n',gtlabels_4mers{unique_code_index},mat2str(groundtruth_codes(unique_code_index,:)));
end

gtlabels = gtlabels_4mers;

% gtlabels{25} = [];
% The Rgs5 barcode was just one color, so we change it's barcode to be
% something impossible to actually align to.
groundtruth_codes(25,:) = [-1 -1 -1 -1];

%% For the case of F2, which was processed as an intact Z-stack, we 
% crop the voxels by Z range by values noted below

if strcmp(params.FILE_BASENAME,'splintr3F2')
    puncta_voxels_filtered  = puncta_indices_cell_filtered{1};
    N = length(puncta_voxels_filtered);
    puncta_centroids_filtered = zeros(N,3);
    
    for p = 1:N
        [x,y,z] = ind2sub(IMG_SIZE,puncta_voxels_filtered{p});
        puncta_centroids_filtered(p,:) = mean([x,y,z],1);
    end
    
    % The morphology was cropped from 131-230 -Dan
    withinZRange = find(puncta_centroids_filtered(:,3)>131 & ...
        puncta_centroids_filtered(:,3)<230);
    
    for rnd_idx = 1:readlength
        puncta_indices_cell_filtered{rnd_idx} = puncta_indices_cell_filtered{rnd_idx}(withinZRange);
        puncta_set_cell_filtered{rnd_idx} = puncta_set_cell_filtered{rnd_idx}(withinZRange,:);
    end
    
    fprintf('Of %i puncta in Splint3F2, %i are within the crop\n',N,...
        length(withinZRange))
    
    puncta_indices_cell = puncta_indices_cell_filtered;
    puncta_set_cell = puncta_set_cell_filtered;
end



%% Unwrap all the puncta into gigantic vectors for quantile norming

%Get the number of filtered puncta
N = length(puncta_indices_cell{1});

funnel_numbers(1) = N;


%Do all the basecalling etc. from one script that has simply been moved out
%of this
%To do the standard color normalization
normalization_qnorm1;
%To do round x color normalization, we call this norm2
% normalization_qnorm2; %This proved to be not as good, 


%N is redefined to be the number of filtered puncta
funnel_numbers(2) = N;


%% Basecalling - simply comparing the normalized intensities
% 
readlength = size(groundtruth_codes,2);

transcript_objects = {};
insitu_transcripts_aligned = [];
match_ctr=1;

shuffled_hits = 0;
not_aligned = zeros(size(insitu_transcripts_filtered,1),1);
for t = 1:size(insitu_transcripts_filtered,1)
    img_transcript = insitu_transcripts_filtered(t,:);
    %Column-shuffling randomization
    %         img_transcript = diag(insitu_transcripts_filtered(randperm(size(insitu_transcripts_filtered,1),4),[1 2 3 4]))';
    
    perfect_match = find(sum(groundtruth_codes == img_transcript,2)==readlength);
    
    % There can only be a unique match in this case
    if length(perfect_match)==1
        transcript = struct;
        transcript.img_transcript=img_transcript;
        
        voxels = puncta_voxels_filtered{t};
        centroid = puncta_centroids_filtered(t,:);
        
        %We then need to modify the indices to be shifted by 130 in Z
        %This is only for SplintrF2
        if strcmp(params.FILE_BASENAME, 'splintr3F2')
            [x,y,z] = ind2sub(IMG_SIZE,voxels);
            z=z-130; %shift into the range of 1-100
            out_of_coords = z<1 |z>100;
            x(out_of_coords) = []; y(out_of_coords) = []; z(out_of_coords) = [];
            voxels = sub2ind(IMG_SIZE,x,y,z);
            centroid = mean([x,y,z],1); 
        end
        
        transcript.pos = centroid;
        transcript.voxels = voxels;
        
        transcript.name = gtlabels{perfect_match};

        transcript_objects{match_ctr} = transcript;
        
        
        % Create an aligned version of the insitu_transcript
        insitu_transcripts_aligned(match_ctr,:) = img_transcript;
        match_ctr = match_ctr+1;
    
    else
        not_aligned(t) = 1;
    end
    
    %Shuffle transcripts to get a false pos rate. Column wise shuffling:
    img_transcript_shuffled = diag(insitu_transcripts_filtered(randperm(size(insitu_transcripts_filtered,1),4),[1 2 3 4]))';
    
    perfect_match = find(sum(groundtruth_codes == img_transcript_shuffled,2)==readlength);
    if length(perfect_match)==1
        shuffled_hits = shuffled_hits+1;
    end
end

% Get all the gene names so we can make a (sorted!) histogram
insitu_genes = cell(length(transcript_objects),1);
for t = 1:length(transcript_objects)
    insitu_genes{t} = transcript_objects{t}.name;
end
insitu_genes = categorical(insitu_genes);
figure; histogram(insitu_genes,'DisplayOrder','descend')
title(sprintf('%i alignments',match_ctr-1));

fprintf('Of %i transcripts, %i matches\n',size(insitu_transcripts_filtered,1),length(transcript_objects));

% Note the 
funnel_numbers(3) = length(transcript_objects); 
funnel_numbers(4) = shuffled_hits;

save(sprintf('%s_results.mat',params.FILE_BASENAME),'transcript_objects');

%% Get statistics on the unalignables

insitu_transcripts_unalignables = insitu_transcripts_filtered(logical(not_aligned),:);

insitu_transcripts_unalignables_string = cell(size(insitu_transcripts_unalignables,1),1);
for t = 1:length(insitu_transcripts_unalignables_string)
    insitu_transcripts_unalignables_string{t} = mat2str(insitu_transcripts_unalignables(t,:));
end
figure
histogram(categorical(insitu_transcripts_unalignables_string),'DisplayOrder','descend')
title(sprintf('%i Unalignables',size(insitu_transcripts_unalignables,1)));

%% Making Gridplots for the puncta

% Note: Does N refer to _filtered or full?
%Currently it should be full

%If we don't close the gridplot figure object each time, the code slows way
%down
figure(2); close


%Get the bounds of the image
N = length(puncta_set_normalized{1});
%Get the total number of voxels involved for the puncta

MIN_QUALITY_THRESHOLD = 5;

% This was manually set for something that would show the 4x4 grid well
figsize = [560   528   560   420];
plotcolors = {'red','magenta','green','blue'};

% Create a random set of 20 puncta (but that is consistent across re-runs
% of the same code) for sampling throughout the 
rng(5); vizlist = randperm(N,20); %1:N

%Looking for all 0-distance options
pindices_to_visualize = zeros(N,1); %[97 257 217 198 152 148 109];
for p_idx = vizlist
    
    %If we made it, note it!
    pindices_to_visualize(p_idx) = 1;
    
    %Get the 1D indices for the puncta in space
    indices = puncta_indices_cell{1}{p_idx};
    
    colormap gray; axis off
    
    figure(2);
    
    set(gcf,'Position',figsize)
    [x,y,z] = ind2sub(IMG_SIZE,indices);
    bnds_x_min = min(x);
    bnds_x_max = max(x);
    bnds_y_min = min(y);
    bnds_y_max = max(y);
    bnds_z_min = min(z);
    bnds_z_max = max(z);
    
    num_voxels = length(indices);
    
    %Puncta signal is x,y,z,rnd,c
    size_x = bnds_x_max-bnds_x_min;%+1;
    size_y = bnds_y_max-bnds_y_min;%+1;
    size_z = bnds_z_max-bnds_z_min;%+1;
    puncta_signal = zeros(size_x,size_y,size_z,readlength,4);
    
    %Map the original voxels per channel into the small subregion that we
    %visualize in the gridplots
    for i = 1:num_voxels
        x_pos = x(i) - bnds_x_min+1;
        y_pos = y(i) - bnds_y_min+1;
        z_pos = z(i) - bnds_z_min+1;
        
        for rnd_idx = 1:readlength
            for c_idx = 1:4
                puncta_signal(x_pos,y_pos,z_pos,rnd_idx,c_idx) = ...
                    puncta_set_normalized{rnd_idx}{p_idx,c_idx}(i);
            end
        end
    end
    
    ha = tight_subplot(readlength,4,zeros(readlength,2)+.01);
    for rnd_idx = 1:readlength
        img4D =  squeeze(puncta_signal(:,:,:,rnd_idx,:));
        
        %Turn the minium value to zero, needed if there are negative zscore
        %values --Deprecated but doesn't hurt anything--
        if min(img4D(:))<0
            %Turn any padding pixels to the minimum value (only if we're
            %zscoring)
            img4D(img4D==0) = min(img4D(:));
            img4D = img4D - min(img4D(:));
        end
        
        %Take the Z projection to visualize in2D
        img4D_maxproj = squeeze(max(img4D,[],3));
        
        %The clims sets the min and max of the imageshow histogram,
        %This was 
        clims = squeeze(clims_perround(:,:,rnd_idx));
        
        for c_idx = 1:4
            %The 2D index of the subplot to show (rows=rounds, cols=colors)
            subplot_idx = (rnd_idx-1)*4+c_idx;
            axes(ha(subplot_idx)); %specific call using the tight_subplot()
            
            img4D_maxproj_onechan =zeros(size(img4D_maxproj));
            img4D_maxproj_onechan(:,:,c_idx) = img4D_maxproj(:,:,c_idx);
            
            top_val = prctile(img4D_maxproj_onechan(:),99);
            
            %Add an additional parameter to indicate illumina color scheme
            rgb = makeRGBImageFrom4ChanData(img4D_maxproj_onechan,clims,true);
            
            imagesc(img4D_maxproj(:,:,c_idx),clims(c_idx,:)); 
            %Note: In a previous version of the code, you can use imshow in
            %place of imagesc, which gives the different colors of the
            %different columns. However, it's hard to compare round
            %intensity, so now we just use imagesc
            colormap gray; axis off
            
            %Show the normalized 99% percentile intensity in the channel's
            %color
            title(sprintf('\\color{%s}%.2f',plotcolors{c_idx},top_val));
            
        end
        
    end
    
    figure(2);
    
    saveas(gcf,fullfile(RESULTSDIR,'gridplots_qnorm1_deconv',sprintf('%03i_puncta.png',p_idx)));
    
    close
    
    
end

fprintf('Done\n');

%% Make a barplot of the funnel to show # puncta -> # reads

figure;
bar(funnel_numbers)
set(gca, 'XTickLabel',funnel_names, 'XTick',1:numel(funnel_names))
xtickangle(45);
text(1:length(funnel_numbers),funnel_numbers,num2str(funnel_numbers),'vert','bottom','horiz','center');
box off

%Plot the base percentage of colors per round
plotBasePercentage(insitu_transcripts_filtered,1:readlength);


%% The way to validate this is to see it!
% Load the registered morphology
exit

if ~exist('imgM','var')
    imgM = load3DTif_uint16(sprintf('%s/%s_round005_chMSHIFT_affine.tif',params.FILE_BASENAME,params.FILE_BASENAME));
    imgM_MIP = max(imgM,[],3);
end

imgPuncta = zeros(size(imgM));
maxPixel = max(imgM(:));
text_pos = [];
text_gene = {};

z_pos = zeros(length(transcript_objects),1);

is_specific_gene = ones(length(transcript_objects),1);
for t_idx = 1:length(transcript_objects)
   
    imgPuncta(transcript_objects{t_idx}.voxels) = maxPixel;
    text_pos(end+1,:) = transcript_objects{t_idx}.pos([1 2]);
    text_gene{end+1} = transcript_objects{t_idx}.name;
    
    %Put in this line if we want to view a specific gene
    %is_specific_gene(t_idx) = strcmp(transcript_objects{t_idx}.name,'Arc');
    
    z_pos(t_idx) = round(transcript_objects{t_idx}.pos(3));
end

RGBZ = zeros([size(imgM,1) size(imgM,2) 3 size(imgM,3)]);

for z = 1:size(imgM,3)
    indices = find(z_pos==z & is_specific_gene);
    
    if isempty(indices)
        continue
    end
    
    
    RGBZ(:,:,:,z) = insertText(RGBZ(:,:,:,z), [text_pos(indices,2),text_pos(indices,1)],...
        text_gene(indices),'textColor',[255 255 255],'BoxColor','blue');
    RGBZ(:,:,3,z) = RGBZ(:,:,3,z)*maxPixel; %Scale up the text from 0-1
    
    fprintf('Completed Z = %i\n',z);
end

RGBZ(:,:,2,:) = imgM;
RGBZ(:,:,1,:) = imgPuncta;

layers = {'Rpuncta','Gmorph','Btext'};
for chan_idx = 1:3
    filename = sprintf('%s/%s_processed_%s.tif',...
        params.FILE_BASENAME,params.FILE_BASENAME,layers{chan_idx});
    save3DTif_uint16(squeeze(RGBZ(:,:,chan_idx,:)),filename);
end

%% Redo above but limit to only puncta co-localized with GFP as defined by 
% a manual threshold

GFPTHRESHOLD = 50;

imgPuncta = zeros(size(imgM));
maxPixel = max(imgM(:));
text_pos = [];
text_gene = {};

z_pos = zeros(length(transcript_objects),1);

is_inmorphology = ones(length(transcript_objects),1);
for t_idx = 1:length(transcript_objects)
   
    imgPuncta(transcript_objects{t_idx}.voxels) = maxPixel;
    text_pos(end+1,:) = transcript_objects{t_idx}.pos([1 2]);
    text_gene{end+1} = transcript_objects{t_idx}.name;
    
    %Put in this line if we want to view a specific gene
    is_inmorphology(t_idx) = prctile(imgM(voxels),99)>GFPTHRESHOLD;
    
    z_pos(t_idx) = round(transcript_objects{t_idx}.pos(3));
end

RGBZ = zeros([size(imgM,1) size(imgM,2) 3 size(imgM,3)]);

for z = 1:size(imgM,3)
    indices = find(z_pos==z & is_inmorphology);
    
    if isempty(indices)
        continue
    end
    
    
    RGBZ(:,:,:,z) = insertText(RGBZ(:,:,:,z), [text_pos(indices,2),text_pos(indices,1)],...
        text_gene(indices),'textColor',[255 255 255],'BoxColor','blue');
    RGBZ(:,:,3,z) = RGBZ(:,:,3,z)*maxPixel; %Scale up the text from 0-1
    
    fprintf('Completed Z = %i\n',z);
end

RGBZ(:,:,2,:) = imgM;
RGBZ(:,:,1,:) = imgPuncta;

layers = {'Rpuncta','Gmorph','Btext'};
for chan_idx = 1:3
    filename = sprintf('%s/%s_processedfiltered_%s.tif',...
        params.FILE_BASENAME,params.FILE_BASENAME,layers{chan_idx});
    save3DTif_uint16(squeeze(RGBZ(:,:,chan_idx,:)),filename);
end
