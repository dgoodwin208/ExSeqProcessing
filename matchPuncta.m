function [ match_vector,distances ] = matchPuncta(puncta_rndA,puncta_rndB)
%This matches round B to round A
[IDX,D] = knnsearch(puncta_rndA,puncta_rndB,'K',5); %getting five other options

match_vector = zeros(size(puncta_rndA,1),1);

%create the distance matrix that is populated by IDX and D
A = sparse(size(puncta_rndB,1),size(puncta_rndA,1));
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

%confusing but ref_idx is the puncta index in the reference round
for matched_row_idx = 1:length(matched_indices_moving)
    
    %Get the indices of the puncta that were matched using the
    %bipartite graph match
    matched_punctaB_idx = matched_indices_moving(matched_row_idx);
    punctaA_idx = matched_indices_ref(matched_row_idx);
    
    
    match_vector(punctaA_idx) = matched_punctaB_idx;
    %Going back to the A matrix (which is indexed like the transpose)
    %to get the original distance value out (has to be re-inverted)
    distances(punctaA_idx) = 1/A(matched_punctaB_idx,punctaA_idx);
    
%     %What if the nearest match is too far away?
%     if distances(punctaA_idx)>DISTANCE_THRESHOLD
%         distances(punctaA_idx) = 0;
%     end
end


end

