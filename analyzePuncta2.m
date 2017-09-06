loadParameters;

load(fullfile(params.punctaSubvolumeDir ,sprintf('%s_puncta_allexp.mat',params.FILE_BASENAME)));

punctamap_master = cell(20,1);
DISTANCE_THRESHOLD = 10;
REF_IDX = 5;

puncta_ref = puncta{REF_IDX}; 
puncta_ref(:,3) = puncta_ref(:,3)*(.5/.165);
puncta_ref = removeRedundantPuncta(puncta_ref);

for rnd_idx = 1:params.NUM_ROUNDS
    
    if rnd_idx==REF_IDX
        continue;
    end
    
    
    puncta_mov = puncta{rnd_idx}; 
    puncta_mov(:,3) = puncta_mov(:,3)*(.5/.165);
    puncta_mov = removeRedundantPuncta(puncta_mov);
    puncta{rnd_idx} = puncta_mov; %Also save back the non-redundant data
 
    %init the holder for this round
    punctamaps = {};
        
    %returns an my-by-1 vector D containing the distances between each
    %observation in Y and the corresponding closest observation in X.
    %That is, D(i) is the distance between X(IDX(i),:) and Y(i,:)
    % [IDX,D] = knnsearch(X,Y,'K',3);
    [IDX,D] = knnsearch(puncta_ref,puncta_mov,'K',1); %getting two other neighbor options
    %So puncta_mov(IDX(i),:) -> puncta_ref(i,:)
    %Note that IDX(i) can be a non-unique value
    
    %For any X's that are more than DISTANCE_THRESHOLD from Ys, remove now
    %This is what removes the reference puncta from consideration
    % distance_filter = D(:,1)>DISTANCE_THRESHOLD;
    % reference_filter_indices = 1:size(Y,1);
    % reference_filter_indices(distance_filter) = [];
    %
    output_ctr = 1;
    % ctr = 1;
    % punctamaps = {};
    
    for ref_idx = 1:size(puncta_ref,1)
        indices_of_MOV = find(IDX == ref_idx);
        
        if isempty(indices_of_MOV)
            %        fprintf('Skipping due to no matches \n');
            continue
        end
        
        distances_to_REF = D(indices_of_MOV);
        
        punctamap = struct;
        punctamap.REF_IDX = 5;
        punctamap.ROUND_IDX = 2;
        punctamap.pos_ref = puncta_ref(ref_idx,:);
        punctamap.index_ref = ref_idx;
        
        if length(indices_of_MOV) == 1
            punctamap.pos_moving = puncta_mov(indices_of_MOV(1),:);
            punctamap.index_mov = indices_of_MOV(1);
            punctamap.distance = distances_to_REF(1);
            punctamap.neighbors = 1;
        else
            [distances_sorted,I] = sort(distances_to_REF,'ascend');
            punctamap.pos_moving = puncta_mov(indices_of_MOV(I(1)),:);
            punctamap.distance = distances_to_REF(1);
            punctamap.index_mov = indices_of_MOV(I(1));
            punctamap.neighbors = numel(distances_to_REF);
        end
        
        if punctamap.distance>DISTANCE_THRESHOLD
            %         fprintf('Skipping due to closest distance: %f\n',punctamap.distance);
            continue;
        end
        
        punctamaps{output_ctr} = punctamap;
        output_ctr= output_ctr+1;
        
        if mod(ref_idx,1000)==0
            fprintf('%i/%i\n',ref_idx,size(puncta_ref,1));
        end
    end
    
    punctamap_master{rnd_idx} = punctamaps;
    
end


votes_refidx = zeros(size(puncta{REF_IDX},1),20);
for rnd_idx = 1:params.NUM_ROUNDS
    
    punctamap = punctamap_master{rnd_idx};
    
    for nn_idx = 1:length(punctamap)
        %votes refidx is indexed the same as the reference round
        votes_refidx(punctamap{nn_idx}.index_ref,rnd_idx)=1;
    end
    
end

total_counts_per_ref = sum(votes_refidx,2);

MIN_NEIGHBOR_AGREEMENT = 17;

REF_INDICES_TO_KEEP = find(total_counts_per_ref>=MIN_NEIGHBOR_AGREEMENT);

%Create an array of all the centroid locations across rounds for a single
%puncta
centroid_collections = zeros(length(REF_INDICES_TO_KEEP),params.NUM_ROUNDS,3);
puncta_roll_call = zeros(length(REF_INDICES_TO_KEEP),params.NUM_ROUNDS);

ctr_progress = 0;

%For each puncta that pass
for ref_puncta_idx = squeeze(REF_INDICES_TO_KEEP)'
    
    rounds_that_need_averaging = ones(params.NUM_ROUNDS,1);
    rounds_that_need_averaging(REF_IDX) = 0;
    
    
    noted_refLoc = 0;
    for rnd_idx = 1:params.NUM_ROUNDS
        
        if rnd_idx==REF_IDX
            continue;
        end
        
        punctamap = punctamap_master{rnd_idx};
        
        %find the idx of the punctamap object that matches
        puncta_obj = struct;

        for nn_idx = 1:length(punctamap)
            if punctamap{nn_idx}.index_ref == ref_puncta_idx
                puncta_obj = punctamap{nn_idx};
                rounds_that_need_averaging(rnd_idx)=0;
                break;
            end
        end
        
        if rounds_that_need_averaging(rnd_idx)
            fprintf('Puncta %i not present in round %i\n',ref_puncta_idx, rnd_idx);
            continue;
        end
        
        centroid_collections(ref_puncta_idx,rnd_idx,:) = puncta_obj.pos_moving;

        %flag that we've noted the reference locatino for this index        
        if ~noted_refLoc
            centroid_collections(ref_puncta_idx,REF_IDX,:) = puncta_obj.pos_ref;
            noted_refLoc=1;	
        end
    end
    %For the rounds that did not have a puncta match to the ref round
    %take the average of the rounds that did have a match
    centroid_collections(ref_puncta_idx,logical(rounds_that_need_averaging),:) = ...
            repmat(...
            mean(squeeze(centroid_collections(ref_puncta_idx,~logical(rounds_that_need_averaging),:)),1), ...
            sum(rounds_that_need_averaging),1);
        
     puncta_roll_call(ref_puncta_idx,:) = rounds_that_need_averaging;
    
ctr_progress = ctr_progress+1;
    
    if mod(ctr_progress,100)==0
       fprintf('Processed %i of %i puncta\n',ctr_progress, length(REF_INDICES_TO_KEEP));
    end
end

save(fullfile(params.punctaSubvolumeDir,sprintf('%s_centroid_collections.mat',params.FILE_BASENAME)),'centroid_collections','-v7.3');


save(fullfile(params.punctaSubvolumeDir,sprintf('%s_punctamap.mat',params.FILE_BASENAME)),'punctamap_master','-v7.3');
