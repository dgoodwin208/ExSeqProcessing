loadParameters;
%Load puncta_centroids, puncta_voxels
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids);

%Load ground truth:
filename_groundtruth = '/Users/Goody/Neuro/ExSeq/simulator/images/simseqtryone_groundtruth_pos+transcripts.mat';
load(filename_groundtruth);
%%
rnd_num = 5;

%Get the locations that the punctafeinder found
centroids_per_round = puncta_centroids{rnd_num};
voxels_per_round = puncta_voxels{rnd_num};

%Ground truth is:
%puncta_covs are the dimensions of each puncta
%puncta_pos are the positions of all puncta

scatter3(puncta_pos(:,1),puncta_pos(:,2),puncta_pos(:,3),'bo');
hold on;
scatter3(centroids_per_round(:,2),centroids_per_round(:,1),centroids_per_round(:,3),'rx');
hold off
legend('Ground Truth','Discovered Puncta');
title(sprintf('%i ground truth puncta vs %i discovered puncta',size(puncta_pos,1),size(centroids_per_round,1)));

%% Use the same knn search from filterPunctaMakeSubvolumes
mov_idx = rnd_num;
DISTANCE_THRESHOLD=10;

puncta_mov = centroids_per_round(:,[2 1 3]); %discovered in img data
puncta_ref = puncta_pos; %Ground truth
% puncta_ref = centroids_per_round(:,[2 1 3]); %discovered in img data
% puncta_mov = puncta_pos; %Ground truth

%init the holder for this round
puncta_matches = zeros(size(puncta_ref,1),params.NUM_ROUNDS);
puncta_matches_distances = zeros(size(puncta_ref,1),params.NUM_ROUNDS)-1;

%returns an my-by-1 vector D containing the distances between each
%observation in Y and the corresponding closest observation in X.
%That is, D(i) is the distance between X(IDX(i),:) and Y(i,:)
% [IDX,D] = knnsearch(X,Y,'K',3);
[IDX,D] = knnsearch(puncta_ref,puncta_mov,'K',1); %getting two other neighbor options
%So puncta_mov(IDX(i),:) -> puncta_ref(i,:)
%Note that IDX(i) can be a non-unique value

%Right now we're doing the most naive way, allowing a puncta from the
%reference round to potentially hit the same moving-round puncta. fine.
output_ctr = 1;
num_discarded_noneighbors = 0;
num_discarded_distance = 0;

%confusing but ref_idx is the puncta index in the reference round
for ref_idx = 1:size(puncta_ref,1)
    indices_of_MOV = find(IDX == ref_idx);
    
    if isempty(indices_of_MOV)
        %fprintf('Skipping due to no matches to ref_idx=%i \n',ref_idx);
        num_discarded_noneighbors = num_discarded_noneighbors+1;
        continue
    end
    
    distances_to_REF = D(indices_of_MOV);
    
    puncta_matches(ref_idx,mov_idx) = 0;
    
    if length(indices_of_MOV) == 1
        puncta_matches(ref_idx,mov_idx) = indices_of_MOV(1);
        puncta_matches_distances(ref_idx,mov_idx) = distances_to_REF(1);
        %             punctamap.pos_moving = puncta_mov(indices_of_MOV(1),:);
        %             punctamap.index_mov = indices_of_MOV(1);
        %             punctamap.distance = distances_to_REF(1);
        %             punctamap.neighbors = 1;
    else
        [distances_sorted,I] = sort(distances_to_REF,'ascend');
        %             punctamap.pos_moving = puncta_mov(indices_of_MOV(I(1)),:);
        %             punctamap.distance = distances_to_REF(1);
        %             punctamap.index_mov = indices_of_MOV(I(1));
        %             punctamap.neighbors = numel(distances_to_REF);
        puncta_matches(ref_idx,mov_idx) = indices_of_MOV(I(1));
        puncta_matches_distances(ref_idx,mov_idx) = distances_sorted(1);
        
    end
    
    if puncta_matches_distances(ref_idx,mov_idx)>DISTANCE_THRESHOLD
        puncta_matches_distances(ref_idx,mov_idx) = 0;
        num_discarded_distance= num_discarded_distance+1;
    end
end

histogram(puncta_matches_distances(:,rnd_num),20)
title('Histogram of distances of ref=groundTruth and mov=sim round 5');
xlabel('Distances in pixels')
ylabel('Count');

%% Get the indices of the reference/groundtruth puncta that DIDN'T Match

matchless_indices = puncta_matches_distances(:,rnd_num)<0;

scatter3(puncta_pos(:,1),puncta_pos(:,2),puncta_pos(:,3),'bo');
hold on;
scatter3(centroids_per_round(:,2),centroids_per_round(:,1),centroids_per_round(:,3),'rx');
scatter3(puncta_pos(matchless_indices,1),puncta_pos(matchless_indices,2),puncta_pos(matchless_indices,3),'g*')
hold off
legend('Ground Truth','Discovered Puncta','Matchless Ground Truth');
title(sprintf('%i ground truth puncta vs %i discovered puncta vs %i unmatched groundtruth',size(puncta_pos,1),size(centroids_per_round,1),sum(matchless_indices)));

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