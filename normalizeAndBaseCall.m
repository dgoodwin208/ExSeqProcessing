loadParameters;

load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));
REF_IDX = params.REFERENCE_ROUND_PUNCTA;

%%

%The normalization method is now: 
%For each color in each expeirmental round,
%subtract the minimum pixel value, 
%then calculate + divide by the mean

puncta_minimums = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
puncta_means = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
puncta_stds = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
for exp_idx = 1:params.NUM_ROUNDS
    
    for c_idx = params.COLOR_VEC
        %Get all the pixels from all the puncta for a round and color
        chan_col = reshape(puncta_set(:,:,:,exp_idx,c_idx,:),[],1);

        %Because we know that these subvolumes have already been masked, we
        %can get that mask back via removing zero entries
        non_zero_mask = chan_col>0;
        chan_col_masked = chan_col(non_zero_mask);
        
        puncta_minimums(exp_idx,c_idx) = min(chan_col_masked);        
        
        chan_col_minshift = chan_col_masked - puncta_minimums(exp_idx,c_idx);
        
        puncta_means(exp_idx,c_idx) = mean(chan_col_minshift);
        puncta_stds(exp_idx,c_idx) = std(chan_col_minshift);
    
    end


exp_idx
end

%%
transcripts = zeros(size(puncta_set,6),params.NUM_ROUNDS);
transcripts_confidence = zeros(size(puncta_set,6),params.NUM_ROUNDS);
puncta_ctr = 1;

number_of_puncta = size(puncta_set,6);

for puncta_idx = 1:number_of_puncta
    
    answer_vector = zeros(params.NUM_ROUNDS,1);
    confidence_vector = zeros(params.NUM_ROUNDS,1);
%     maxes = zeros(4,1);
    for exp_idx = 1:params.NUM_ROUNDS
        
        %Load a 10x10x10 x NUM_CHANNELS
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));

        %Get the non-zero pixels from the subvolume 
        normalized_puncta_vector = cell(params.NUM_CHANNELS,1);
        
        %Get the mask across all rounds:
        punctaset_maxproj_on_channels = max(punctaset_perround,[],4);
        puncta_mask_linear  = punctaset_maxproj_on_channels(:)>0;
        
%         barf()
        
        normed_mean_values = zeros(4,1);
        for c_idx = 1:params.NUM_CHANNELS
            %load and linearize the pixels
            pixels_from_subvolume = reshape(punctaset_perround(:,:,:,c_idx),1,[]);
            pixels_masked = pixels_from_subvolume(puncta_mask_linear);

            %Removing minimum value then dividing by the mean
%             normalized_puncta_vector{c_idx} = ...
%                 (pixels_masked - puncta_minimums(exp_idx,c_idx))/puncta_means(exp_idx,c_idx);
    %Standard Z-score method: subtract min, subtract mean, divide by std.
            normalized_puncta_vector{c_idx} = ...
                (pixels_masked - puncta_minimums(exp_idx,c_idx) - puncta_means(exp_idx,c_idx))/puncta_stds(exp_idx,c_idx);

            normed_mean_values(c_idx) = mean(normalized_puncta_vector{c_idx});
        end
        
        [values,indices] = sort(normed_mean_values,'descend');
       
        answer_vector(exp_idx) = indices(1);
        
        confidence_vector(exp_idx) = values(1)/(values(1)+values(2));
        
    end
    
    transcripts(puncta_ctr,:) = answer_vector;
    transcripts_confidence(puncta_ctr,:) = confidence_vector;

    %TEMP: just get it to run
    pos_for_reference_round(puncta_ctr,:) = pos(:,params.REFERENCE_ROUND_PUNCTA,puncta_idx);
%     pos_for_reference_round(puncta_ctr,:) = zeros(1,3);
    puncta_ctr = puncta_ctr +1;
    
    fprintf('Calling base puncta #%i out of %i \n',puncta_ctr, size(puncta_set,6));
    
    
end

save(fullfile(params.punctaSubvolumeDir,'transcriptsv13_punctameannormed.mat'),'transcripts','transcripts_confidence','pos_for_reference_round');

