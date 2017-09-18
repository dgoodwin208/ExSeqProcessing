loadParameters;

load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));
REF_IDX = params.REFERENCE_ROUND_PUNCTA;

%%
puncta_set_normed = zeros(size(puncta_set));

%The normalization method is now: 
%For each color in each expeirmental round,
%subtract the minimum pixel value, 
%then calculate + divide by the mean
for exp_idx = 1:params.NUM_ROUNDS
    clear chan_col
    for c = params.COLOR_VEC
        chan_col(:,c) = reshape(puncta_set(:,:,:,exp_idx,c,:),[],1);
    end

    chan_col = chan_col - repmat(min(chan_col,[],1),size(chan_col,1),1);
    mean_vec = mean(chan_col,1);
    cols_normed = chan_col ./ repmat(mean_vec,size(chan_col,1),1);

    for c = params.COLOR_VEC
        puncta_set_normed(:,:,:,exp_idx,c,:) = reshape(cols_normed(:,c),squeeze(size(puncta_set(:,:,:,exp_idx,c,:))));
    end

exp_idx
end


transcripts = zeros(size(puncta_set_normed,6),params.NUM_ROUNDS);
transcripts_confidence = zeros(size(puncta_set_normed,6),params.NUM_ROUNDS);
puncta_ctr = 1;


for puncta_idx = 1:size(puncta_set_normed,6)
    
    answer_vector = zeros(params.NUM_ROUNDS,1);
    confidence_vector = zeros(params.NUM_ROUNDS,1);
    maxes = zeros(4,1);
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set_normed(:,:,:,exp_idx,:,puncta_idx));
        
        max_intensity = max(punctaset_perround(:))+1;
        min_intensity = min(punctaset_perround(:));
        
        [max_chan, confidence] = chooseChannel(punctaset_perround,params.COLOR_VEC,params.DISTANCE_FROM_CENTER);
        
        %Check that we're not comparing four empty channels (which is the
        %case when there was not a suitable puncta found in this round)
        if max_intensity>1
            answer_vector(exp_idx) = max_chan;
        else
            answer_vector(exp_idx) = 0;
        end
        confidence_vector(exp_idx) = confidence;
        
    end
    
    transcripts(puncta_ctr,:) = answer_vector;
    transcripts_confidence(puncta_ctr,:) = confidence_vector;

    %TEMP: just get it to run
    pos_for_reference_round(puncta_ctr,:) = pos(:,REF_IDX,puncta_idx);
%     pos_for_reference_round(puncta_ctr,:) = zeros(1,3);
    puncta_ctr = puncta_ctr +1;
    
    fprintf('Calling base puncta #%i out of %i \n',puncta_ctr, size(puncta_set_normed,6));
    
    
end

save(fullfile(params.punctaSubvolumeDir,'transcriptsv13_punctameannormed.mat'),'transcripts','transcripts_confidence','pos_for_reference_round');

