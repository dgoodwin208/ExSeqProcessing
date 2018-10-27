function [ merge_flags ] = calculateMergeFlags( puncta_centroids,MERGE_DISTANCE)
%calculateMergeFlags For a cell array of centroids per round and a distance
%for which it's likely that two (or more) puncta could get merged if the
%colors were the same, give each puncta a number for the possible merge
%errors around it

num_rounds = length(puncta_centroids);
merge_flags = cell(num_rounds,1);

for rnd_idx = 1:num_rounds
    
    puncta_rndA = puncta_centroids{rnd_idx};
    
    merge_flags_per_round = zeros(size(puncta_rndA,1),1);
    
    %For each puncta, find it's nearest neighbor in the same round
    [IDX,D] = knnsearch(puncta_rndA,puncta_rndA,'K',5); %getting four other options
   
    %For each puncta, ignore the mapping to itself, and note the number of
    %possible merge mistakes for this puncta
    for puncta_idx = 1:size(puncta_rndA,1)
        
        %Find the indices of the D and IDX cols that point to puncta that
        %are within the MERGE_DISTANCE AND not the same puncta
        indices_merge_candidates = (IDX(puncta_idx,:) ~= puncta_idx) & ...
            (D(puncta_idx,:)<=MERGE_DISTANCE);
        
        merge_flags_per_round(puncta_idx) = sum(indices_merge_candidates);
    end
    merge_flags{rnd_idx} = merge_flags_per_round;
end

end

