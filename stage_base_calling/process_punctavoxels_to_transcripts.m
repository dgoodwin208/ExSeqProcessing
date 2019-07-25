

%Load the set of all pixels of all puncta across all rounds from the
filename_pixels = sprintf(fullfile(params.punctaSubvolumeDir,sprintf('%s_punctavoxels.mat',params.FILE_BASENAME)));

load(filename_pixels)

fprintf('Loaded.\n');

%This is the size of the cropped images, corresponding to ~15um pre-ExM
filename_punctaVol = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
vol = load3DImage_uint16(filename_punctaVol);
IMG_SIZE = size(vol);
clear vol filename_punctaVol

%We're reading 4-base barcodes
readlength = 4;

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
%for unique_code_index = 1:num_unique_barcodes
%    fprintf('%s\t%s\n',gtlabels_4mers{unique_code_index},mat2str(groundtruth_codes(unique_code_index,:)));
%end

gtlabels = gtlabels_4mers;

% gtlabels{25} = [];
% The Rgs5 barcode was just one color, so we change it's barcode to be
% something impossible to actually align to.
groundtruth_codes(25,:) = [-1 -1 -1 -1];

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

save(fullfile(params.basecallingResultsDir,sprintf('%s_results.mat',params.FILE_BASENAME)),'transcript_objects','funnel_numbers');
