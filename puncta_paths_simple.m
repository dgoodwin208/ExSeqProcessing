% loadParameters;

%This loads puncta_centroids and puncta_voxels (the list of voxel INDICES)
%per puncta
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids)

load('groundtruth_dictionary.mat')

THRESHOLD_MATCHING_DISTANCE = 10;
MERGE_DISTANCE = 5;    

filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,6,'ch00'));
sample_img = load3DTif_uint16(filename_in);
img_size = size(sample_img);



%%

sum_img = zeros(img_size);
for round_num = 3:params.NUM_ROUNDS
    fprintf('Round %i\n',round_num);

     for i= 1:length(puncta_voxels{round_num})
            %Set the pixel value to somethign non-zero
        indices_to_add = puncta_voxels{round_num}{i};
        sum_img(indices_to_add)=sum_img(indices_to_add)+1;
     end
end

filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_ALLPixelIdxListSummedNoPrimer.tif',params.FILE_BASENAME));
save3DTif_uint16(sum_img,filename_out);


%%

stack_in = sum_img;
bw = stack_in>5; %MAGIC NUMBER MINIMUM NUMBER OF ROUNDS!
D = bwdist(~stack_in);
D = -D;
D(~bw) = Inf;
L = watershed(D);
L(~bw) = 0;

candidate_puncta= regionprops(L,stack_in, 'WeightedCentroid', 'PixelIdxList');
indices_to_remove = [];
for i= 1:length(candidate_puncta)
    if size(candidate_puncta(i).PixelIdxList,1)< params.PUNCTA_SIZE_THRESHOLD
        indices_to_remove = [indices_to_remove i];
    end
end

good_indices = 1:length(candidate_puncta);
good_indices(indices_to_remove) = [];

filtered_puncta = candidate_puncta(good_indices);
fprintf('Removed %i candidate puncta for being too small\n',...
    length(candidate_puncta)-length(filtered_puncta));

centroids_temp = zeros(length(filtered_puncta),3);
voxels_temp = cell(length(filtered_puncta),1);
for p_idx = 1:length(filtered_puncta)
    centroids_temp(p_idx,:) = filtered_puncta(p_idx).WeightedCentroid;
    voxels_temp{p_idx} = filtered_puncta(p_idx).PixelIdxList;
end


%Create a new set of puncta_centroids and puncta_voxels based ONLY
%on the aggregate puncta poitns
puncta_centroids = cell(params.NUM_ROUNDS,1);
puncta_voxels = cell(params.NUM_ROUNDS,1);

for rnd_idx = 1:params.NUM_ROUNDS
    puncta_centroids{rnd_idx} = centroids_temp;
    puncta_voxels{rnd_idx} = voxels_temp;
end


%% Examine the nearest neighbor of every pixel and flag it as a possible
% merge error in another round
%make a holder vector of the sizes of all
num_puncta_per_round = zeros(params.NUM_ROUNDS,1);
for rnd_idx = 1:params.NUM_ROUNDS
    num_puncta_per_round(rnd_idx) = size(puncta_centroids{rnd_idx},1);
end

for demerge_iter = 1:1
    
    %For each round, calculate a set of merge flags which indicate the
    %number of puncta that are near enough that they could be incorrectly
    %merged if the expressed channels are the same
    merge_flags = calculateMergeFlags(puncta_centroids,MERGE_DISTANCE);
    
    %--------Collect all possible puncta paths from the perfect matching    
    all_possible_punctapaths = zeros(sum(num_puncta_per_round),params.NUM_ROUNDS);
    
    %The row is the reference round, the col is the moving round
    %A is row
    demerge_indices = cell(params.NUM_ROUNDS,1);
    delete_indices = cell(params.NUM_ROUNDS,1);
    for rnd_idx_A = 1:params.NUM_ROUNDS
        
        fprintf('Finding puncta paths for ref_round=%i\n',rnd_idx_A);
        row_offset = sum(num_puncta_per_round(1:(rnd_idx_A-1)));
        
        punctaA = puncta_centroids{rnd_idx_A};
        demerge_indices_per_round = zeros(size(punctaA,1),1);
        %B is column
        for rnd_idx_B = 1:params.NUM_ROUNDS
            %When the moving = ref, it's just the identity
            if rnd_idx_B == rnd_idx_A
                for rndA_puncta_idx = 1:size(punctaA,1)
                    all_possible_punctapaths(row_offset+rndA_puncta_idx,rnd_idx_B)=rndA_puncta_idx;
                end
                continue;
            end
            punctaB = puncta_centroids{rnd_idx_B};
            
            %Output of matchPuncta is the size of punctaA. If there are no
            %matches, the index is zero. There is also no distance filtering,
            %so that has to be done here.
            [matches_AB,distances_AB] = matchPuncta(punctaA,punctaB,puncta_voxels{rnd_idx_A},puncta_voxels{rnd_idx_B});
            
            for match_idx = 1:length(matches_AB) %which is also size of (punctaA,1)
                if (matches_AB(match_idx)==0) 
                    %Leave that entry blank if the perfect matching came back
                    %with nothing, or if the distances to a match is too far
                    all_possible_punctapaths(row_offset+match_idx,rnd_idx_B) = -1;
                else
                    all_possible_punctapaths(row_offset+match_idx,rnd_idx_B)=matches_AB(match_idx);
                end
            end
        end
        
        fprintf('\tSearching for possible merge errors...');
        %Question: For all the punctaB we just matched with punctaA as the reference,
        %How many punctaA centroids are less than the mode of the merge
        %flag?
        
        %Get the list of all the punctapaths we just generated
        possible_puncta_path_subset = all_possible_punctapaths((row_offset+1):(row_offset+length(matches_AB)),:);
        
        %Get rid of any puncta path with a missing puncta in it.
        possible_puncta_path_subset_filtered = possible_puncta_path_subset(...
            sum(possible_puncta_path_subset==-1,2)==0,:);
        
        %Now, for the rnd_idx A reference location, check for any case where the
        %merge flag of rnd_idx A is less than the mode of its matches
        merge_flag_rndA = merge_flags{rnd_idx_A};
        
        for pp_idx = 1:size(possible_puncta_path_subset_filtered,1)
            puncta_indices_per_round = possible_puncta_path_subset_filtered(pp_idx,:);
            
            merge_flags_for_puncta_path = zeros(params.NUM_ROUNDS,1);
            for rnd_idx =1:params.NUM_ROUNDS
                merge_flags_for_puncta_path(rnd_idx) = merge_flags{rnd_idx}(puncta_indices_per_round(rnd_idx));
            end
            merge_flag_rndA = merge_flags_for_puncta_path(rnd_idx_A);
            
            %If the merge flag for rndA is less than the mode of the merge
            %flags in the other rounds connected to rndAs puncta, then it's
            %likely rndA is a merge problem and is flagged for duplication.
            if merge_flag_rndA < mode(merge_flags_for_puncta_path)
                demerge_indices_per_round(puncta_indices_per_round(rnd_idx_A))=1;
            end
        end
        
        fprintf('Found %i\n',sum(demerge_indices_per_round));
        demerge_indices{rnd_idx_A} = demerge_indices_per_round;
        
        %Maybe later we could also create a list of puncta to remove from
        %consideration if they are very clearly extraneous. In this case, if
        %the mode of the puncta path is -1.
    end
    
    % ----- For all the de-merged indices, duplicate them!
    
    for rnd_idx = 1:params.NUM_ROUNDS
        current_round_centroids = puncta_centroids{rnd_idx};
        current_round_centroids_baseguess = puncta_baseguess{rnd_idx};
        current_round_voxels = puncta_voxels{rnd_idx};
        current_round_demerge_flags = demerge_indices{rnd_idx};
        extra_centroids = [];new_centroids_ctr = 1;
        extra_centroids_baseguess = [];
        extra_centroids_voxels = {};
        for p_idx = 1:size(current_round_centroids,1)
            if logical(current_round_demerge_flags(p_idx))
                extra_centroids(new_centroids_ctr,:) = current_round_centroids(p_idx,:);
                extra_centroids_baseguess(new_centroids_ctr) = current_round_centroids_baseguess(p_idx);
                extra_centroids_voxels{new_centroids_ctr} = current_round_voxels{p_idx};
                new_centroids_ctr = new_centroids_ctr+1;
            end
        end
        fprintf('Generated %i duplicates into round %i\n',new_centroids_ctr-1,rnd_idx);
        
        %Concatenate the additional puncta to the end of puncta_centroids!
        puncta_centroids{rnd_idx} = [current_round_centroids;extra_centroids]; 
        puncta_baseguess{rnd_idx} = [current_round_centroids_baseguess;extra_centroids_baseguess'];
        puncta_voxels{rnd_idx} = [current_round_voxels;extra_centroids_voxels'];
    end
    
end %end demerge iterations
%% Find the paths that match to all rounds, then check merge flags

%Filter to make sure a puncta is present in every round 
%somewhat unclear logic below but it works. The <1 term can be modified if
%you want to allow paths with missing puncta
filtered_paths = all_possible_punctapaths(sum(all_possible_punctapaths==-1,2)<1,:);

%First we re-map all the duplicated puncta to handle the merge errors back
%into the original indieces
all_possible_punctapaths_demerged = filtered_paths;

%% Analyze how well these transcripts align!

%To save time of matching identical transcripts, first find all the unique
%puncta paths throughout the data
[unique_paths, ia, ic] = unique(all_possible_punctapaths_demerged,'rows');

%% Generate all unique paths by votes

%First generate a vote vector for all unique paths
%Then, starting from the top-voted paths, sequentially remove candidate paths 
%that use puncta that have already been used
%See how many times each unique path is present in the original set of all
%possible possible paths
unique_paths_votes = zeros(size(unique_paths,1),1);
for path_idx = 1: size(unique_paths,1)
    
    %Find out how many times each puncta index in each rond matches the
    %unique
    hits = (all_possible_punctapaths_demerged==unique_paths(path_idx,:));
    %Logical array of matches, then sum the rows to find the mefect match
    %Ignore primer rounds
    votes = sum(hits(:,4:end),2)==(params.NUM_ROUNDS-3);
    unique_paths_votes(path_idx) = sum(votes);
end

%% Now step progressively down the votes and create a set of mutually
%exclusive paths

[votescore,~] = sort(unique(unique_paths_votes),'descend');
exclusive_paths = []; excl_ctr = 1;
exclusive_paths_votes = [];
pathoptions = logical(ones(size(unique_paths,1),1));

%Starting from all the top scores
for v = votescore'
   indices_for_score = find(unique_paths_votes==v);
   
   %Ties might exist for, say, de-merged puncta
   %for now we simply take the first one we come across 
   for vote_score_idx = indices_for_score'
    
       %Only consider path options that have not been removed due to a
       %puncta that has already been used
       if ~pathoptions(vote_score_idx)
           continue;
       end
       
       %Add to the exclusive path
       new_exclusive_path = unique_paths(vote_score_idx,:);
       
       exclusive_paths(excl_ctr,:) = new_exclusive_path;
       exclusive_paths_votes(excl_ctr) = v;
       excl_ctr = excl_ctr+1;
       
       %For any row in uniquepaths that has any puncta that we've already
       %used, mark it as unavailable by AND operator
       pathoptions = pathoptions & ~any(unique_paths==new_exclusive_path,2);
   end
end

%% 
%Remove those pesky 0s for now, leftover from the more stringent distance
%metric addition
exclusive_paths_votes(any(exclusive_paths==0,2)) = [];
exclusive_paths(any(exclusive_paths==0,2),:) = [];

exclusive_paths_transcripts = zeros(size(exclusive_paths));
exclusive_path_positions = zeros(size(exclusive_paths_transcripts,1),3);
for excl_path_idx = 1:size(exclusive_paths_transcripts,1)
   
    positions_across_rounds = zeros(params.NUM_ROUNDS,3);
    
    for rnd_idx = 1:params.NUM_ROUNDS
       
        %As a quick starting point use the punctafeinder base calling,
        %which is naive but works pretty well
%         exclusive_paths_transcripts(excl_path_idx,rnd_idx) = puncta_baseguess{rnd_idx}(exclusive_paths(excl_path_idx,rnd_idx));
    
        %Puncta_centroids is the output of the punctafeinder
        %filtered_unique_paths is a row vector where each round is the
        %index of that puncta in a round
        pos = puncta_centroids{rnd_idx}(exclusive_paths(excl_path_idx,rnd_idx),:);
        positions_across_rounds(rnd_idx,:) = pos;
        
    end

    exclusive_path_positions(excl_path_idx,:) = mean(positions_across_rounds,1); 
    
end

final_positions = exclusive_path_positions;
% final_transcripts = exclusive_paths_transcripts;
final_punctapaths = exclusive_paths;
final_votes = exclusive_paths_votes;

%%

filename_output = fullfile(params.punctaSubvolumeDir,sprintf('%s_finalmatches.mat',params.FILE_BASENAME));
save(filename_output,'final_positions',...
    'final_punctapaths','final_votes','all_possible_punctapaths_demerged');

filename_centroidsMOD = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels_demerged.mat',params.FILE_BASENAME));
save(filename_centroidsMOD,'puncta_centroids','puncta_voxels','puncta_baseguess');
