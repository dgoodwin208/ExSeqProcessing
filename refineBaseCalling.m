% Load transcriptsv6 and rois_votednonnormed16b

%If it's an old .mat file, be sure to refine the puncta indices
if size(puncta_set,6)==num_puncta
   puncta_set = puncta_set(:,:,:,:,:,good_puncta_indices);
    Y = Y(good_puncta_indices);
    X = X(good_puncta_indices);
    Z = Z(good_puncta_indices); 
end
%Transcriptsv6 has been called on quantile normalized puncta per round
%The puncta_set_normed data is not stored  and puncta_set
loadParameters;

%Create vectors in a cell array that will take all raw pixels from which we
%will create the distributions
raw_pixels = cell(params.NUM_ROUNDS,params.NUM_CHANNELS,2);
%Note that the third index is as follows:
%1 = background
%2 = signal
IDX_BACKGROUND =1;
IDX_SIGNAL = 2;
%Initialize each entry as a list
for chan_idx = 1:params.NUM_CHANNELS
    for exp_idx = 1:params.NUM_ROUNDS
        raw_pixels{exp_idx,chan_idx,1} = [];
        raw_pixels{exp_idx,chan_idx,2} = [];
    end
end

%Now loop through all puncta (non-normalized) and, depending on which was
%called by the normalized comparison (in normalizePunctaVector.m) put the
%center 5x6x6 pixels into the vectors that will create the distirbution
% central_puncta_indices = ...
%     ceil(params.PUNCTA_SIZE/2 - params.DISTANCE_FROM_CENTER):ceil(params.PUNCTA_SIZE/2 + params.DISTANCE_FROM_CENTER);
central_puncta_indices= 5:6;
for puncta_idx = 1:size(puncta_set,6)
    for exp_idx = 1:params.NUM_ROUNDS
        %Get which channel was called for a puncta and round
        winning_index = transcripts(puncta_idx,exp_idx);
        background_indices = setdiff(params.COLOR_VEC,winning_index);
        
        % For the winner, get the central 6x6x6 volume, linearize it 
        % and add into the vector
        subvolume = puncta_set(central_puncta_indices,...
            central_puncta_indices,...
            central_puncta_indices,...
            exp_idx,...
            winning_index, puncta_idx);
        raw_pixels{exp_idx,winning_index,IDX_SIGNAL} = ...
            [raw_pixels{exp_idx,winning_index,IDX_SIGNAL}; subvolume(:)];
        
        %For the other background rounds, add the linearized subvolume 
        %to the respective vecotrs
        for other_index = background_indices
            subvolume = puncta_set(central_puncta_indices,...
                central_puncta_indices,...
                central_puncta_indices,...
                exp_idx,...
                other_index, puncta_idx);
            raw_pixels{exp_idx,other_index,IDX_BACKGROUND} = ...
                [raw_pixels{exp_idx,other_index,IDX_BACKGROUND}; subvolume(:)];
        end
    end
    if mod(puncta_idx,100)==0
        fprintf('Processed puncta %i/%i\n',puncta_idx,size(puncta_set,6));
    end
end

%% Let's look at some histograms!

NUM_BUCKETS = 500;
for exp_idx = 1:3
    figure;
   
    subplot(params.NUM_CHANNELS,1,params.NUM_CHANNELS);
    for chan_idx = [1,2,4]
        subplot(params.NUM_CHANNELS,1,chan_idx);
        %Load all the raw pixels 
        chanvec_bg = raw_pixels{exp_idx,chan_idx,IDX_BACKGROUND};
        chanvec_sig = raw_pixels{exp_idx,chan_idx,IDX_SIGNAL};

        %Remove the top 1% of data so we can visualize cleaner histograms
        percentiles_bg  = prctile(chanvec_bg,[0,99]);
        percentiles_sig = prctile(chanvec_sig,[0,99]);
        %Instead of deleting we'll instead just cap the value to the 99%
        outlierIndex_bg = chanvec_bg > percentiles_bg(2);        
        chanvec_bg(outlierIndex_bg) = percentiles_bg(2); 
        outlierIndex_sig = chanvec_sig > percentiles_sig(2);        
        chanvec_sig(outlierIndex_sig) = percentiles_sig(2);
        
        fprintf('Removed %.03f and %.03f outliers for bg and sig, respectively\n',...
            sum(outlierIndex_bg)/length(raw_pixels{exp_idx,chan_idx,IDX_BACKGROUND}),...
            sum(outlierIndex_sig)/length(raw_pixels{exp_idx,chan_idx,IDX_SIGNAL}));
       
        
        %Concatenate the two so we can get proper bcuke edges
        [values,binedges] = histcounts([chanvec_bg; chanvec_sig],NUM_BUCKETS);
        
        [values_bg,binedges_bg] = histcounts(chanvec_bg,binedges);
        [values_sig,binedges_sig] = histcounts(chanvec_sig,binedges);
        
        
        b = bar(binedges(1:NUM_BUCKETS),values_bg,'b');
        b.FaceAlpha = 0.3;
        hold on;        
        b = bar(binedges(1:NUM_BUCKETS),values_sig,'r');
        b.FaceAlpha = 0.3;
        hold off;
        title(sprintf('Experiment %i, Color %i',exp_idx, chan_idx));
        legend('Background','Signal');
    end
end
hold off;
