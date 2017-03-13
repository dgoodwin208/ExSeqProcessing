% Load transcriptsv7 and rois_votednonnormed16b

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

%Create mean and min values for each channel and each round
min_values = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
mean_values = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
for exp_idx = 1:params.NUM_ROUNDS
    clear chan_col
    chan_col(:,1) = reshape(puncta_set(:,:,:,exp_idx,1,:),[],1);
    chan_col(:,2) = reshape(puncta_set(:,:,:,exp_idx,2,:),[],1);
    chan_col(:,4) = reshape(puncta_set(:,:,:,exp_idx,4,:),[],1);
    
    min_values(exp_idx,:) = min(chan_col,[],1);
    %Take the mean after subtracting the min
    mean_values(exp_idx,:) = mean(chan_col - repmat(min_values(exp_idx,:),size(chan_col,1),1),1);
    
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
        %process the subvolume to subtract the mean and divi
        subvolume = (subvolume(:) - min_values(exp_idx,winning_index))...
            /mean_values(exp_idx,winning_index);
        raw_pixels{exp_idx,winning_index,IDX_SIGNAL} = ...
            [raw_pixels{exp_idx,winning_index,IDX_SIGNAL}; subvolume];
        
        %For the other background rounds, add the linearized subvolume
        %to the respective vecotrs
        for other_index = background_indices
            subvolume = puncta_set(central_puncta_indices,...
                central_puncta_indices,...
                central_puncta_indices,...
                exp_idx,...
                other_index, puncta_idx);
            subvolume = (subvolume(:) - min_values(exp_idx,other_index))...
                /mean_values(exp_idx,other_index);
            raw_pixels{exp_idx,other_index,IDX_BACKGROUND} = ...
                [raw_pixels{exp_idx,other_index,IDX_BACKGROUND}; subvolume];
        end
    end
    if mod(puncta_idx,100)==0
        fprintf('Processed puncta %i/%i\n',puncta_idx,size(puncta_set,6));
    end
end

%% Let's look at some histograms!

for exp_idx = 1:3
    figure;
    
    subplot(params.NUM_CHANNELS,1,params.NUM_CHANNELS);
    for chan_idx = params.COLOR_VEC %[1,2,4]
        subplot(params.NUM_CHANNELS,1,chan_idx);
        %Load all the raw pixels
        chanvec_bg = raw_pixels{exp_idx,chan_idx,IDX_BACKGROUND};
        chanvec_sig = raw_pixels{exp_idx,chan_idx,IDX_SIGNAL};
        
        %Remove the top 1% of data so we can visualize cleaner histograms
        %         percentiles_bg  = prctile(chanvec_bg,[0,99]);
        %         percentiles_sig = prctile(chanvec_sig,[0,99]);
        %         %Instead of deleting we'll instead just cap the value to the 99%
        %         outlierIndex_bg = chanvec_bg > percentiles_bg(2);
        %         chanvec_bg(outlierIndex_bg) = percentiles_bg(2);
        %         outlierIndex_sig = chanvec_sig > percentiles_sig(2);
        %         chanvec_sig(outlierIndex_sig) = percentiles_sig(2);
        %
        %         fprintf('Removed %.03f and %.03f outliers for bg and sig, respectively\n',...
        %             sum(outlierIndex_bg)/length(raw_pixels{exp_idx,chan_idx,IDX_BACKGROUND}),...
        %             sum(outlierIndex_sig)/length(raw_pixels{exp_idx,chan_idx,IDX_SIGNAL}));
        
        
        
        %Concatenate the two so we can get proper bucket edges
        [values,binedges] = histcounts([chanvec_bg; chanvec_sig],params.NUM_BUCKETS);
        
        [values_bg,binedges_bg] = histcounts(chanvec_bg,binedges);
        [values_sig,binedges_sig] = histcounts(chanvec_sig,binedges);
        
        
        b = bar(binedges(1:params.NUM_BUCKETS),values_bg,'b');
        b.FaceAlpha = 0.3;
        hold on;
        b = bar(binedges(1:params.NUM_BUCKETS),values_sig,'r');
        b.FaceAlpha = 0.3;
        hold off;
        title(sprintf('Experiment %i, Color %i',exp_idx, chan_idx));
        legend('Background','Signal');
    end
end
hold off;

%% Making a distribution for every color, round

%Each probability will be an Nx2 vector, containing both the center of the
%histogram edges and the probability
%so it's a 5D vector...
probabilities = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,params.NUM_BUCKETS,2,2);
IDX_HIST_VALUES = 1;
IDX_HIST_BINS = 2;
for exp_idx = 1:3
    
    for chan_idx = params.COLOR_VEC %[1,2,4]
%         subplot(params.NUM_CHANNELS,1,chan_idx);
        %Load all the raw pixels
        chanvec_bg = raw_pixels{exp_idx,chan_idx,IDX_BACKGROUND};
        chanvec_sig = raw_pixels{exp_idx,chan_idx,IDX_SIGNAL};
        
        %Make the binedges by concaneating both distributions from this
        %channel
        [values,binedges] = histcounts([chanvec_bg; chanvec_sig],params.NUM_BUCKETS);
        
        %Then use our createEmpDistributionFromVector function to make the
        %distributions
        [p,b] = createEmpDistributionFromVector(chanvec_bg,binedges);
        probabilities(exp_idx,chan_idx,:,IDX_BACKGROUND,IDX_HIST_VALUES) = p;
        probabilities(exp_idx,chan_idx,:,IDX_BACKGROUND,IDX_HIST_BINS) = b;
        [p,b] = createEmpDistributionFromVector(chanvec_sig,binedges);
        probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_VALUES) = p;
        probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_BINS) = b;
    end
end

%% Plot all probabilities for a sanity check
figure;
subplot(params.NUM_ROUNDS,length(params.COLOR_VEC),1);
ctr = 1;
for exp_idx = 1:params.NUM_ROUNDS
    for chan_idx = params.COLOR_VEC
        subplot(params.NUM_ROUNDS,length(params.COLOR_VEC),ctr);
        plot(squeeze(probabilities(exp_idx,chan_idx,:,IDX_BACKGROUND,IDX_HIST_BINS)),squeeze(probabilities(exp_idx,chan_idx,:,IDX_BACKGROUND,IDX_HIST_VALUES)));
        hold on;
        plot(squeeze(probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_BINS)),squeeze(probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_VALUES)))
        hold off;
        ctr = ctr +1;
        
        title(sprintf('Exp%i, Chan%i',exp_idx,chan_idx))
    end
end
%% Now with the probabilities matrix, every puncta in every color in every
% round can have a probability calculated of the log likelihood that the
% puncta is drawn from the signal or the background distribution

central_puncta_indices= 5:6;

%Do this for one experiment
prob_transcripts = zeros(size(puncta_set,6),params.NUM_ROUNDS, params.NUM_CHANNELS);
for puncta_idx = 1:size(puncta_set,6)
    for exp_idx = 1:params.NUM_ROUNDS
        for chan_idx = params.COLOR_VEC
            
            
            % Get the central pixels to query the joint distribution of their
            % occurance
            subvolume = puncta_set(central_puncta_indices,...
                central_puncta_indices,...
                central_puncta_indices,...
                exp_idx,...
                chan_idx, puncta_idx);
            
            % Use the pre-calcualated min and mean values to normalize the
            % puncta
            subvolume = (subvolume(:) - min_values(exp_idx,chan_idx))...
                /mean_values(exp_idx,chan_idx);
            
            p_sig = squeeze(probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_VALUES));
            p_bg = squeeze(probabilities(exp_idx,chan_idx,:,IDX_BACKGROUND,IDX_HIST_VALUES));
            
            b_sig = squeeze(probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_BINS));
            b_bg = squeeze(probabilities(exp_idx,chan_idx,:,IDX_BACKGROUND,IDX_HIST_BINS));
            
            
            jprob_sig = calculateJointProbability(subvolume,p_sig,b_sig);
            jprob_bg = calculateJointProbability(subvolume,p_bg,b_bg);
            prob_transcripts(puncta_idx,exp_idx,chan_idx) = jprob_bg/jprob_sig;
        end
    end
    
    if mod(puncta_idx,100)==0
        fprintf('Processed %i \n',puncta_idx)
    end
end

%% Correct way of calculating probabilistic transcripts
% Need the comparative transcripts too for this
central_puncta_indices= 5:6;

%Do this for one experiment
prob_transcripts = zeros(size(puncta_set,6),params.NUM_ROUNDS, params.NUM_CHANNELS);
for exp_idx = 1:params.NUM_ROUNDS
    % Make color priors for the channels per experiment
    color_prior = zeros(1,params.NUM_CHANNELS);
    for chan_idx = params.COLOR_VEC
        color_prior(chan_idx) = sum(transcripts(:,exp_idx)==chan_idx)/...
            size(transcripts,1);
    end
    
    for puncta_idx = 1:size(puncta_set,6)        
        for chan_idx = params.COLOR_VEC

            % Get the central pixels to query the joint distribution of their
            % occurance
            subvolume = puncta_set(central_puncta_indices,...
                central_puncta_indices,...
                central_puncta_indices,...
                exp_idx,...
                chan_idx, puncta_idx);
            % Use the pre-calcualated min and mean values to normalize the
            % puncta
            subvolume = (subvolume(:) - min_values(exp_idx,chan_idx))...
                /mean_values(exp_idx,chan_idx);
            
            p_sig = squeeze(probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_VALUES));
            b_sig = squeeze(probabilities(exp_idx,chan_idx,:,IDX_SIGNAL,IDX_HIST_BINS));
            
            %calculateJointProbability creates sum of log probabilities
            jprob_sig = calculateJointProbability(max(subvolume),p_sig,b_sig);
            
            other_indices = setdiff(params.COLOR_VEC,chan_idx);
            probabilities_of_backgrounds = zeros(1,length(other_indices));
            ctr = 1;
            for other_index = other_indices
                subvolume = puncta_set(central_puncta_indices,...
                    central_puncta_indices,...
                    central_puncta_indices,...
                    exp_idx,...
                    other_index, puncta_idx);
                % Use the pre-calcualated min and mean values to normalize the
                % puncta
                subvolume = (subvolume(:) - min_values(exp_idx,other_index))...
                    /mean_values(exp_idx,other_index);
            
                p_bg = squeeze(probabilities(exp_idx,other_index,:,IDX_BACKGROUND,IDX_HIST_VALUES));
                b_bg = squeeze(probabilities(exp_idx,other_index,:,IDX_BACKGROUND,IDX_HIST_BINS));
                
                probabilities_of_backgrounds(ctr) = calculateJointProbability(max(subvolume),p_bg,b_bg);
                ctr = ctr +1;
            end
            
            prob_transcripts(puncta_idx,exp_idx,chan_idx) = ...
                log(color_prior(chan_idx)) + sum(probabilities_of_backgrounds)...
                + sum(jprob_sig);
        end
        
        if mod(puncta_idx,100)==0
            fprintf('Processed puncta %i for exp %i \n',puncta_idx, exp_idx)
        end
    end

end

%Just for now, make all of channel 3 -Inf
prob_transcripts(:,:,3) = -Inf;

%%

for exp_idx = 1:3
    figure;
    plot3(squeeze(prob_transcripts(:,exp_idx,1)),...
        squeeze(prob_transcripts(:,exp_idx,2)),...
        squeeze(prob_transcripts(:,exp_idx,4)),'.');
    title(sprintf('Scatter plot of probalistic base calling', exp_idx));
    grid on;
    title(sprintf('Exp%i Comparison of 3 color channels internal probabilities',exp_idx));
end

%% Comparing to other transcript
%prob_transcripts is just for experiment 1, so transcripts(:,1)
agreements = zeros(size(puncta_set,6),3);
for exp_idx = 1:params.NUM_ROUNDS
    [~,prob_calls] = max(squeeze(prob_transcripts(:,exp_idx,:)),[],2);
    
    agreements(:,exp_idx) = prob_calls == transcripts(:,exp_idx);
    sum(agreements(:,exp_idx))
    transcripts_probsolo(:,exp_idx) = prob_calls;
    
    %Confidence here will simply be the raw probability value
    for puncta_idx = 1:size(puncta_set,6)
        transcripts_probsolo_confidence(puncta_idx,exp_idx) = ...
            squeeze(prob_transcripts(puncta_idx,exp_idx,prob_calls(puncta_idx))); 
    end
    
end
%How many puncta agree on ALL THREE?
indices_interAndIntraAgreements = all(agreements,2);
sum(indices_interAndIntraAgreements)

transcripts_probfiltered = transcripts(indices_interAndIntraAgreements,:);
transcripts_probfiltered_confidence = transcripts_confidence(indices_interAndIntraAgreements,:);
transcripts_probfiltered_probconfidence = exp(transcripts_probsolo_confidence(indices_interAndIntraAgreements,:));

