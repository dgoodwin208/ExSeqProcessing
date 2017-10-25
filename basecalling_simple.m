
if ~exist('puncta_set','var')
    load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));
end

%% Convert all the data into zscores (very cheap base calling)
puncta_set_normed = zeros(size(puncta_set));
for c = params.COLOR_VEC
    chan_col(:,c) = reshape(puncta_set(:,:,:,:,c,:),[],1);
end

% cols_normed = quantilenorm(chan_col);
cols_normed = zscore(chan_col);

for c = params.COLOR_VEC
    puncta_set_normed(:,:,:,:,c,:) = reshape(cols_normed(:,c),size(squeeze(puncta_set(:,:,:,:,c,:))));
end


%% Compare the top 10 brightest (zscore) pixels in each puncta across channels

path_indices = 1:size(final_positions,1);

%Pre-initialize the cell arrray and determine the basecalls
total_number_of_pixels = 0;

chans = zeros(params.PUNCTA_SIZE^3,4);

for p_idx= 1:length(path_indices) %accidentally deleted whatever was here
    
    path_idx = path_indices(p_idx);
    
    for rnd_idx = 1:params.NUM_ROUNDS
        
        %Load and vectorize the puncta_subset
        for c = 1:params.NUM_CHANNELS
            chantemp = puncta_set_normed(:,:,:,rnd_idx,c,p_idx);
            chans(:,c) = chantemp(:)';
        end
        
        sorted_chans = sort(chans,1,'descend');
        %Take the mean of the top 10 values
        scores = mean(sorted_chans(1:10,:),1);
        %and the new baseguess 
        [~, newbaseguess] = max(scores);
        base_calls_quickzscore(p_idx,rnd_idx) = newbaseguess;
       
        total_number_of_pixels = total_number_of_pixels + 100;
    end
end

[unique_transcipts,~,~] = unique(base_calls_quickzscore,'rows');
fprintf('Found %i transcripts, %i of which are duplicates\n',size(base_calls_quickzscore,1),size(unique_transcipts,1));

fprintf('Scoring...');
consideration_mask = logical([0 0 0 1 1 ones(1,5) 1 ones(1,9)]);
max_hits = sum(consideration_mask);
final_hammingscores = zeros(size(base_calls_quickzscore,1),1);
for t_idx = 1:size(unique_transcipts,1)
    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes(:,consideration_mask(4:end))==unique_transcipts(t_idx,consideration_mask));
    
    %Calculate the hamming distance
    scores = max_hits- sum(hits,2);
    
    final_hammingscores(t_idx) = min(scores);
end

fprintf('Done!\n');