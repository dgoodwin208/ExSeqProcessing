

%Load the set of all pixels of all puncta across all rounds from the
filename_pixels = sprintf(fullfile(params.punctaSubvolumeDir,sprintf('%s_punctavoxels.mat',params.FILE_BASENAME)));
load(filename_pixels)
fprintf('Loaded Pixels.\n');

%This is the size of the cropped images, corresponding to ~15um pre-ExM
filename_punctaVol = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
%image dimensions is a utility to get the size of the image without loading
%just be careful with the XY parameter switch
IMG_SIZE = image_dimensions(filename_punctaVol);
if strcmp(params.IMAGE_EXT,'tif')
    IMG_SIZE = IMG_SIZE([2,1,3]);
end
clear filename_punctaVol
fprintf('Loaded PunctaMap.\n');

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

%% Load the barcodes
groundtruth_dict = params.GROUND_TRUTH_DICT;
fprintf('Using dictonary %s \n', groundtruth_dict)
load(groundtruth_dict);

%Get the number of filtered puncta
N = length(puncta_indices_cell{1});
funnel_numbers(1) = N;

readlength = params.NUM_ROUNDS;


%% Are there any filtering functions that need to be applied?

if isfield(params, 'BASECALLING_FILTERINGFUNCTION')
    fprintf('Calling custom filtering fuction: %s\n',func2str(params.BASECALLING_FILTERINGFUNCTION))
    params.BASECALLING_FILTERINGFUNCTION();
else
    %If we want to do no experiment-specific filtering, just comment out these
    %two lines:    
    puncta_indices_cell_filtered = puncta_indices_cell;
    puncta_set_cell_filtered = puncta_set_cell;
end

%% Unwrap all the puncta into gigantic vectors for quantile norming




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
base_mismatch_ctr = zeros(1,params.NUM_ROUNDS);
for t = 1:size(insitu_transcripts_filtered,1)
    img_transcript = insitu_transcripts_filtered(t,:);
    %Shuffle transcripts to get a false pos rate. 
    %Column wise shuffling, basicallyodrawing randomly from each base
    img_transcript_shuffled = diag(insitu_transcripts_filtered(randperm(size(insitu_transcripts_filtered,1),readlength),1:readlength))';
    
    match_scores = readlength - sum(groundtruth_codes == img_transcript,2);
    [score, score_idx] = sort(match_scores,'ascend');
    
    %If the minimum score is either above the allowable threshold or
    %non-unique, leave it
    numread_minscore = sum(score==score(1));
    if score(1)<= params.BASECALLING_MAXHAMMING  && numread_minscore==1
        
        transcript = struct;
        transcript.img_transcript=img_transcript;
        transcript.known_sequence_matched = groundtruth_codes(score_idx(1),:);
        
        %monitoring
        base_mismatch_ctr = base_mismatch_ctr + ...
            single(img_transcript ~= groundtruth_codes(score_idx(1),:));
        
        transcript.hamming =score(1);
        voxels = puncta_voxels_filtered{t};
        centroid = puncta_centroids_filtered(t,:);
        
        transcript.pos = centroid;
        %adjust the position by the cropped dimensions from earlier
        %processing
        transcript.pos(1) = transcript.pos(1) + crop_dims(1,1)-1;
        transcript.pos(2) = transcript.pos(2) + crop_dims(2,1)-1;
        transcript.pos(3) = transcript.pos(3) + crop_dims(3,1)-1;
        transcript.voxels = voxels;
        
        transcript.name =gtlabels{score_idx(1)};
        
        transcript.intensity_norm = squeeze(puncta_intensities_norm(t,:,:));
        transcript.intensity_raw = squeeze(puncta_intensities_raw(t,:,:));
        
        transcript_objects{match_ctr} = transcript;
        
        % Create an aligned version of the insitu_transcript
        insitu_transcripts_aligned(match_ctr,:) = img_transcript;
        match_ctr = match_ctr+1;
    
    else
        not_aligned(t) = 1;
    end
    
    
    %Now replicate to get the randomized score
    match_scores = readlength - sum(groundtruth_codes == img_transcript_shuffled,2);
    [score, score_idx] = sort(match_scores,'ascend');
    
    %If the minimum score is either above the allowable threshold or
    %non-unique, leave it
    numread_minscore = sum(score==score(1));
    if score(1)<= params.BASECALLING_MAXHAMMING  && numread_minscore==1
        shuffled_hits = shuffled_hits+1;
    end
end

fprintf('Of %i transcripts, %i matches\n',size(insitu_transcripts_filtered,1),length(transcript_objects));

% Note the 
funnel_numbers(3) = length(transcript_objects); 
funnel_numbers(4) = shuffled_hits;

save(fullfile(params.basecallingResultsDir,sprintf('%s_basecalls.mat',params.FILE_BASENAME)),'insitu_transcripts_filtered','puncta_intensities_norm','puncta_intensities_raw','puncta_centroids_filtered','demixing_matrices','-v7.3');
save(fullfile(params.basecallingResultsDir,sprintf('%s_transcriptobjects.mat',params.FILE_BASENAME)),'transcript_objects','funnel_numbers','crop_dims','-v7.3');


%% Generate bogus data - Totally random data
insitu_transcripts_filtered = randi(4,size(insitu_transcripts,1),7);
%How many zeros are there in the original data?
num_zero_entries = sum(insitu_transcripts(:)==0);
d1_indices = randperm(numel(insitu_transcripts),num_zero_entries);
insitu_transcripts_filtered(d1_indices)=0;
%Col random:

%% Generate bogus data - Column shuffled data
numts = size(insitu_transcripts,1);
for col = 1:readlength
    insitu_transcripts_filtered(:,col) = insitu_transcripts(randperm(numts),col);
end
