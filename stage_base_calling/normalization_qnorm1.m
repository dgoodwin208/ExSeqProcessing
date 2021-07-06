% function [insitu_transcripts_filtered,puncta_voxels_filtered,puncta_centroids_filtered] = ...
%     normalization_qnorm1(puncta_set_cell_filtered)

N = size(puncta_set_cell_filtered{1},1);
readlength = params.NUM_ROUNDS;
% puncta_present = ones(N,readlength);
%Get the total number of voxels involved for the puncta
%used to initialize data_cols
total_voxels = 0;
for n = 1:N
    total_voxels = total_voxels + length(puncta_set_cell_filtered{1}{n});
end

puncta_set_normalized = puncta_set_cell_filtered;

%Create a clims object for each round
%This will be later used for calculating the histogram settings for
%visualizing the puncta gridplots
%The 2 is that we will be noting two percentiles of brightness for determining puncta
clims_perround = zeros(params.NUM_CHANNELS,2,readlength);

%We'll also do some new insitu_transcript base calling while we're at it
insitu_transcripts = zeros(N,readlength);
%Each image is normalized seperately
illumina_corrections = zeros(readlength,1);

puncta_intensities_raw = zeros(N,readlength,params.NUM_CHANNELS);
puncta_intensities_norm = zeros(N,readlength,params.NUM_CHANNELS);
demixing_matrices = zeros(params.NUM_CHANNELS,params.NUM_CHANNELS,readlength);

puncta_hasjunk_rounds = zeros(N,readlength);
for rnd_idx = 1:readlength
    
    %this vector keeps track of where we're placing each set of voxels per
    %color into a vector
    punctaindices_vecpos = zeros(N,1);
    pos_cur = 1;
    
    data_cols = zeros(total_voxels,params.NUM_CHANNELS);
    for p_idx = 1:N
        voxels = puncta_set_cell_filtered{rnd_idx}{p_idx};
        n = length(voxels);
        
        %mark the next spot as we pack in the 1D vectors of the puncta
        pos_next = pos_cur+n-1;
        data_cols(pos_cur:pos_next,1) = puncta_set_cell_filtered{rnd_idx}{p_idx,1};
        data_cols(pos_cur:pos_next,2) = puncta_set_cell_filtered{rnd_idx}{p_idx,2};
        data_cols(pos_cur:pos_next,3) = puncta_set_cell_filtered{rnd_idx}{p_idx,3};
        data_cols(pos_cur:pos_next,4) = puncta_set_cell_filtered{rnd_idx}{p_idx,4};
        
        punctaindices_vecpos(p_idx) = pos_next;
        pos_cur = pos_next+1;
        
    end
    
    %We have to mask out the zero values so they don't disrupt the
    %normalization process, so we get all voxel indices for which all
    %channels are non zero
    %Get the nonzero mask pixel indices
    nonzero_mask = all(data_cols>0,2);
    
    fprintf('Fraction of pixels that are nonzero: %f\n',sum(nonzero_mask)/length(nonzero_mask));
    %Get the subset of data_cols that is nonzero
    data_cols_nonzero = data_cols(nonzero_mask,:);
    
    %Now remove puncta in which more than one of the channels is an extreme
    %outlier
    if isfield(params,'BASECALLING_ARTIFACT_THRESH')
        excess_thresh = params.BASECALLING_ARTIFACT_THRESH;
    else
        excess_thresh = [Inf,Inf,Inf,Inf];
    end
    
    nonjunk_mask = sum(data_cols_nonzero>excess_thresh,2)<params.MAXNUM_MISSINGROUND;
    data_cols_nonzero_nonjunk = data_cols_nonzero(nonjunk_mask,:);
    
    %Everything that is nonzero but junk, mark as nan
    data_cols_nonzero(~nonjunk_mask,:) = nan;
    
    
    %Normalize just the nonzero portion
    if size(data_cols_nonzero_nonjunk,1)>1
        data_cols_norm_nonzero_nonjunk = quantilenorm(data_cols_nonzero_nonjunk);
    else
        data_cols_norm_nonzero_nonjunk = data_cols_nonzero_nonjunk;
    end
    
    %Finally, do we want to attempt to remove covariance between channels?
    if params.BASECALLING_PERFORMWHITENING
        fprintf('Performing whitening\n');
        %Note, what comes out of whiten is the transformed version of the
        %de-meaned data. Since we are whitening the qnormed data, the means
        %should be the same anyway, so for comparison it should be all good
        [data_cols_norm_nonzero_nonjunk, ~, ~, demixing_matrix] = whiten(data_cols_norm_nonzero_nonjunk );
        
    else
        %If you want to to Experiment with removing the demixing:
        data_cols_norm_nonzero_nonjunk = data_cols_norm_nonzero_nonjunk;
        demixing_matrix = eye(4);
    end
    demixing_matrices(:,:,rnd_idx) = demixing_matrix;
    
    %Apply the demixing (either identity or calculated via whitening)
    scaled_demixing_matrix = demixing_matrix./repmat(max(demixing_matrix,[],1),4,1,1);
    
    %Initialize the data_cols_norm as the data_cols
    data_cols_norm_nonzero = data_cols_nonzero; %has the NaNs
    %place the whitened, normalized nonjunk back in
    data_cols_norm_nonzero(nonjunk_mask,:) = data_cols_norm_nonzero_nonjunk ;
    
    %Finally, get the ouput
    data_cols_norm = data_cols;
    %Replace all the nonzero entries with the normed values
    data_cols_norm(nonzero_mask,:) = data_cols_norm_nonzero;
    
    %
    %Unpack the normalized colors back into the original shape
    pos_cur = 1;
    puncta_hasjunk = zeros(N,1);
    for p_idx = 1:N
        puncta_set_normalized{rnd_idx}{p_idx,1} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),1);
        puncta_set_normalized{rnd_idx}{p_idx,2} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),2);
        puncta_set_normalized{rnd_idx}{p_idx,3} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),3);
        puncta_set_normalized{rnd_idx}{p_idx,4} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),4);
        
        %Declare junk if there are any "junk" voxels in any of the channels
        puncta_hasjunk(p_idx) = any(any(isnan(data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),1:4))));
        
        puncta_intensities_norm(p_idx,rnd_idx,:) = prctile(data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),:),params.BASECALLING_SIG_THRESH);
        
        puncta_intensities_raw(p_idx,rnd_idx,:) = prctile(data_cols(pos_cur:punctaindices_vecpos(p_idx),:),params.BASECALLING_SIG_THRESH);
        
        %Track the 1D index as we unpack out each puncta
        pos_cur = punctaindices_vecpos(p_idx)+1;
        
    end
    fprintf('Discarding %i bases over the thresh limit\n',sum(puncta_hasjunk));
    puncta_hasjunk_rounds(:,rnd_idx) = puncta_hasjunk;
    
    
    %Filter out puncta that have a missing round, as determined by the
    %color cutoffs
    for p_idx = 1:N
        
        %We skip any puncta that has a NaN in it, which is enumerated with
        %an NaN
        
        if puncta_hasjunk(p_idx)
            insitu_transcripts(p_idx,rnd_idx) = 0;
            continue
        end
        
        %A channel is present if the brightest pixel in the puncta are
        %brighter than the cutoff we just defined
        %We choose the 90th percentile because we don't want to be subject
        %to outlier pixels that might enter a puncta, yet we also don't
        %want to use the mean which could be dragged down by background.
        % However, using the background subtraction in the puncta
        % extraction phase seems to make us be conservative with the pixels
        % we count, so background is likely not a big factor
        
        
        chan1_signal = puncta_intensities_norm(p_idx,rnd_idx,1);
        chan2_signal = puncta_intensities_norm(p_idx,rnd_idx,2);
        chan3_signal = puncta_intensities_norm(p_idx,rnd_idx,3);
        chan4_signal = puncta_intensities_norm(p_idx,rnd_idx,4);
                
        chan1_present = ~(chan1_signal==0);
        chan2_present = ~(chan2_signal==0);
        chan3_present = ~(chan3_signal==0);
        chan4_present = ~(chan4_signal==0);
        
        %When we do puncta extraction, if all the signal is removed 
        %(ie, less than zero value remaining), we set the value to 1, rather
        %than zero. If a puncta is all 1s, then it was no signal and will 
        %be caught in this check. This logic does not explicitly check for
        %1s but that's the only way four floats would be the same number.
        all_channels_same = all(puncta_intensities_norm(p_idx,rnd_idx,1) == ...
            squeeze(puncta_intensities_norm(p_idx,rnd_idx,:)) );
        
        [signal_strength,winning_base] = max([chan1_signal,chan2_signal,chan3_signal,chan4_signal]);
        
        %In the case that a signal is missing in any channel, we cannot
        %call that base so mark it a zero  
        all_channels_present = chan1_present & chan2_present & ...
            chan3_present & chan4_present & ~all_channels_same;
        if all_channels_present
            insitu_transcripts(p_idx,rnd_idx) = winning_base;
        else
            insitu_transcripts(p_idx,rnd_idx) = 0;
        end
        
        %         puncta_present(p_idx,rnd_idx) = chan1_present & chan2_present & ...
        %             chan3_present & chan4_present;
    end
    
    
    fprintf('Completed Round %i\n',rnd_idx);
end
fprintf('Completed normalization!\n');

puncta_complete = 1:N; %keep everything, don't filter


fprintf('Number of puncta before filtering missing bases: %i\n',N);
fprintf('Number of puncta after filtering missing bases: %i\n',length(puncta_complete));

N = length(puncta_complete);
puncta_set_normalized_filtered = puncta_set_normalized;
for rnd_idx = 1:readlength
    puncta_set_normalized_filtered{rnd_idx} = puncta_set_normalized{rnd_idx}(puncta_complete,:);
    
end
puncta_voxels_filtered = puncta_indices_cell{1}(puncta_complete);

insitu_transcripts_filtered = insitu_transcripts(puncta_complete,:);

puncta_centroids_filtered = zeros(N,3);
IMG_SIZE_CROPPED = [length(crop_dims(1,1):crop_dims(1,2)),...
    length(crop_dims(2,1):crop_dims(2,2)),...
    length(crop_dims(3,1):crop_dims(3,2))];
for p = 1:N
    [x,y,z] = ind2sub(IMG_SIZE_CROPPED,puncta_voxels_filtered{p});
    puncta_centroids_filtered(p,:) = mean([x,y,z],1);
end

