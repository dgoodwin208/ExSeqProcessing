loadParameters;
%Load puncta_centroids, puncta_voxels
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids);

filename_transcripts = fullfile(params.transcriptResultsDir, 'transcript_objects.mat');
load(filename_transcripts);

%Load ground truth:
filename_groundtruth = '/Users/Goody/Neuro/ExSeq/simulator/images/simseqtryone_groundtruth_pos+transcripts.mat';
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

%% Use the same knn+bipartite graph matching from filterPunctaMakeSubvolumes

%Make a vector to store the index of the matching extracted transcript 
groundtruth_matches = zeros(size(puncta_pos,1),1);
%Similar, but init distances as -1 so we know when there's not a match
groundtruth_distances = zeros(size(puncta_pos,1),1)-1;
puncta_ref = puncta_pos;
puncta_mov = img_transcript_positions;
[IDX,D] = knnsearch(puncta_ref,puncta_mov,'K',5); %getting five other options

%create the distance matrix that is populated by IDX and D
A = sparse(size(puncta_mov,1),size(puncta_ref,1));
for idx_row = 1:size(IDX,1)
    for idx_col = 1:size(IDX,2)
        %For that row in the IDX, loop over the columns, which are the
        %indices to the reference puncta round
        %The entries are the inverse of distance, which is useful
        %because we're going to get the maximum weighted partition
        A(idx_row,IDX(idx_row,idx_col)) = 1/D(idx_row,idx_col);
    end
end

%Using a bipartite complete matching algorithm
[~, matched_indices_moving,matched_indices_ref] = bipartite_matching(A);

num_discarded_noneighbors = 0;
num_discarded_distance = 0;

%confusing but ref_idx is the puncta index in the reference round
for matched_row_idx = 1:length(matched_indices_moving)
    
    %Get the indices of the puncta that were matched using the
    %bipartite graph match
    matched_puncta_mov_idx = matched_indices_moving(matched_row_idx);
    ref_puncta_idx = matched_indices_ref(matched_row_idx);
    
    
    groundtruth_matches(ref_puncta_idx) = matched_puncta_mov_idx;
    %Going back to the A matrix (which is indexed like the transpose)
    %to get the original distance value out (has to be re-inverted)
    groundtruth_distances(ref_puncta_idx) = 1/A(matched_puncta_mov_idx,ref_puncta_idx);    
end


figure
histogram(groundtruth_distances,30)
title('Histogram of distances of ref=groundTruth and mov=sim round 5');
xlabel('Distances in pixels')
ylabel('Count');

nonmatched_gt_indices = groundtruth_distances==-1;
%Plot the ground truth puncta that did not match with the extracted puncta
figure;
scatter3(puncta_pos(nonmatched_gt_indices,1),puncta_pos(nonmatched_gt_indices,2),puncta_pos(nonmatched_gt_indices,3),'rx');
hold on;
scatter3(puncta_pos(~nonmatched_gt_indices,1),puncta_pos(~nonmatched_gt_indices,2),puncta_pos(~nonmatched_gt_indices,3),'ko');
hold off
legend('Ground Truth that didnt match','Ground truths that did match');
title(sprintf('%i ground truth puncta that were not tracked',sum(nonmatched_gt_indices)));


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