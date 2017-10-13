% loadParameters;

%Load puncta_centroids, puncta_voxels
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids);

filename_transcripts = fullfile(params.transcriptResultsDir, 'transcript_objects.mat');
load(filename_transcripts);

%Load ground truth:
filename_groundtruth = fullfile(simparams.OUTPUTDIR, 'simseqtryone_groundtruth_pos+transcripts.mat');
load(filename_groundtruth);
%%
rnd_num = params.REFERENCE_ROUND_PUNCTA;

%Get the locations that the punctafeinder found
centroids_per_round = puncta_centroids{rnd_num};
voxels_per_round = puncta_voxels{rnd_num};

%Ground truth is:
%puncta_covs are the dimensions of each puncta
%puncta_pos are the positions of all puncta
%puncta_transcripts are the transcripts
scatter3(puncta_pos(:,1),puncta_pos(:,2),puncta_pos(:,3),'bo');
hold on;
scatter3(centroids_per_round(:,2),centroids_per_round(:,1),centroids_per_round(:,3),'rx');
hold off
legend('Ground Truth','Discovered Puncta');
title(sprintf('%i ground truth puncta vs %i discovered puncta',size(puncta_pos,1),size(centroids_per_round,1)));

%% Match the positions of discovered image transcripts and positions to
%the ground truth positions and

img_transcript_positions = zeros(length(transcript_objects),3);
img_transcript_sequences = zeros(length(transcript_objects),20);
for obj_idx = 1:length(transcript_objects)
    img_transcript_positions(obj_idx,:) = transcript_objects{obj_idx}.pos-padwidth;
    img_transcript_sequences(obj_idx,:) = transcript_objects{obj_idx}.img_transcript;
end

scatter3(puncta_pos(:,1),puncta_pos(:,2),puncta_pos(:,3),'bo');
hold on;
scatter3(img_transcript_positions(:,1),img_transcript_positions(:,2),img_transcript_positions(:,3),'rx');
hold off
legend('Ground Truth','Positions of extracted transcripts');
title(sprintf('%i ground truth puncta vs %i discovered puncta',size(puncta_pos,1),size(centroids_per_round,1)));

%% Match the positions of the extracted puncta to their closes ground truth puncta
%Then compare the codes

%Make a vector to store the index of the matching extracted transcript 
groundtruth_matches = zeros(size(puncta_pos,1),1);
%Similar, but init distances as -1 so we know when there's not a match
groundtruth_distances = zeros(size(puncta_pos,1),1)-1;

puncta_ref = puncta_pos;
puncta_mov = final_positions(:,[2 1 3]);

[IDX,D] = knnsearch(puncta_ref,puncta_mov,'K',1); %getting five other options

num_discarded_noneighbors = 0;
num_discarded_distance = 0;

%confusing but ref_idx is the puncta index in the reference round

hamming_to_nearestRef = zeros(size(final_positions,1),1);
for mov_idx = 1:size(puncta_mov,1)
    
    %Get the indices of the puncta that were matched using the
    %bipartite graph match
    matched_puncta_ref_idx = IDX(mov_idx);
    
    transcripts_gt = puncta_transcripts(matched_puncta_ref_idx,:);
    transcripts_discovered = final_transcripts(mov_idx,:);
    
    hamming_to_nearestRef(mov_idx) = 17-sum(transcripts_gt(4:end)==transcripts_discovered(4:end));
    
end


figure
histogram(hamming_to_nearestRef)
title('Histogram of Hamming distances to the nearest ground truth puncta');
xlabel('Hamming score')
ylabel('Count');

% nonmatched_gt_indices = groundtruth_distances==-1;
% %Plot the ground truth puncta that did not match with the extracted puncta
% figure;
% scatter3(puncta_pos(nonmatched_gt_indices,1),puncta_pos(nonmatched_gt_indices,2),puncta_pos(nonmatched_gt_indices,3),'rx');
% hold on;
% scatter3(puncta_pos(~nonmatched_gt_indices,1),puncta_pos(~nonmatched_gt_indices,2),puncta_pos(~nonmatched_gt_indices,3),'ko');
% hold off
% legend('Ground Truth that didnt match','Ground truths that did match');
% title(sprintf('%i ground truth puncta that were not tracked',sum(nonmatched_gt_indices)));


%% Compare the ground truth of transcripts to their 
final_scores = zeros(size(puncta_pos,1),1)-1;
for gt_match_idx = 1:size(puncta_pos,1)
   %If we know this gt transcript doesn't have a match, skip it
    if nonmatched_gt_indices(gt_match_idx)
        continue; 
    end
   
    gt_transcript = puncta_transcripts(gt_match_idx,:);
    est_transcript = img_transcript_sequences(groundtruth_matches(gt_match_idx),:);
    final_scores(gt_match_idx) = size(gt_transcript,2) - sum(gt_transcript==est_transcript);
end

final_scores_filtered = final_scores(final_scores>=0); 
figure
histogram(final_scores_filtered,unique(final_scores_filtered))
title('Histogram of hamming scores of extracted transcripts vs ground truth transcripts');
xlabel('Distances in pixels')
ylabel('Count');

%% For the puncta that did match, let's compare puncta sizes

matched_indices = puncta_matches_distances(:,rnd_num)>0;

matched_covs = puncta_covs(matched_indices,:);
puncta_sizes_gt = prod(matched_covs*2,2);

mov_indices = puncta_matches(matched_indices,rnd_num);

puncta_sizes_found = zeros(length(mov_indices),1);

for m_idx = 1:length(mov_indices)
    pixels_for_puncta = voxels_per_round{mov_indices(m_idx)};
    puncta_sizes_found(m_idx) = numel(pixels_for_puncta);
end

figure;
scatter(puncta_sizes_gt,puncta_sizes_found);
xlabel('Product of 2*covariance');
ylabel('Number of pixels found from the image');
title('Scatter plot of ground truth puncta parameters and size of detected puncta');

%% How many of these transcripts are in the original data?
%Load ground truth:
if ~exist('puncta_transcripts','var')
    filename_groundtruth = '/Users/Goody/Neuro/ExSeq/simulator/images/simseqtryone_groundtruth_pos+transcripts.mat';
    load(filename_groundtruth);
end

%Ground truth is:
%puncta_covs are the dimensions of each puncta
%puncta_pos are the positions of all puncta
%puncta_transcripts are the transcripts

%Randomize the reference
% puncta_transcripts = puncta_transcripts(:,randperm(20));

best_scores_insample = size(final_transcripts,1);
best_scores_totalref = size(final_transcripts,1);
perfect_transcripts = []; perfect_transcripts_ctr = 1;
% acceptable_paths = zeros(size(filtered_unique_paths,1),1);
for t_idx = 1:size(final_transcripts,1)
    transcript = final_transcripts(t_idx,(4:end));
    
    %-------------------
    %Search for a perfect match in the ground truth of field of view
    hits = (puncta_transcripts(:,4:end)==transcript);

    %Calculate the hamming distance
    scores = length(transcript)- sum(hits,2);
    
    best_scores_insample(t_idx) = min(scores);
    if best_scores_insample(t_idx)<=1
        perfect_transcripts(perfect_transcripts_ctr,:) = transcript;
        perfect_transcripts_ctr = perfect_transcripts_ctr+1;
    end
    
    %-------------------
    %Search for a perfect match in complete reference
    hits = (groundtruth_codes==final_transcripts(t_idx,4:end));
    %Calculate the hamming distance
    scores = length(transcript)- sum(hits,2);
    best_scores_totalref(t_idx) = min(scores);
    
end

fprintf('Done matching!');
figure
subplot(1,2,1)
histogram(best_scores_insample)
title('Histogram of hamming scores to GT in sample');
subplot(1,2,2)
histogram(best_scores_totalref)
title('Histogram of hamming scores to GT in total reference');
