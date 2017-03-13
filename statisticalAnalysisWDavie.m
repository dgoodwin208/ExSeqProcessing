%Assuming puncta_set is loaded and filtered by good_indices

NUM_ROUNDS = 3;
NUM_COLORS = 4;
DISTANCE_FROM_CENTER = 2.5;
puncta_vector = zeros(size(puncta_set,6),NUM_ROUNDS,NUM_COLORS);
for z = 1:size(puncta_vector,1)
    for exp_idx = 1:NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,z));
        
        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        
        [max_chan, confidence,scores] = chooseChannelMod(punctaset_perround,[1,2,4],DISTANCE_FROM_CENTER);
        puncta_vector(z,exp_idx,:) = scores;
    end
    
    if(mod(z,100)==0)
        fprintf('Processing puncta #%i out of %i \n',z, size(puncta_vector,1));
    end
    
end

%% Make distributions

figure;
subplot(3,1,1);
colors = {'r','b','k','g'};
NUM_BUCKETS = 100;
for exp_idx = 1:3
    subplot(3,1,exp_idx);
    for chan_idx = [1,2,4]
        chanvec = squeeze(puncta_vector(:,exp_idx,chan_idx));
        
        %Remove entries larger than 3 stddevs above the mean
        mean_val = mean(chanvec);
        std_val = std(chanvec);
        chanvec(chanvec>mean_val+3*std_val) = mean_val+3*std_val;
        
        [values,binedges] = histcounts(chanvec,NUM_BUCKETS);
        
        
        %         b = bar(binedges(1:NUM_BUCKETS),values,colors{chan_idx});
        b = bar(1:NUM_BUCKETS,values,colors{chan_idx});
        b.FaceAlpha = 0.3;
        hold on;
        
    end
    title(sprintf('Experiment %i',exp_idx));
    xlabel('Histogram bucket indices');
    legend('Chan1','Chan2','Chan4');
end
hold off;

%% Make a distribution for round1, color1

puncta_vector_filtered = puncta_vector(accepted_locations,:,:);
figure;
subplot(3,1,1);
colors = {'r','b','k','g'};
NUM_BUCKETS = 100;
for exp_idx = 1:3
    subplot(3,1,exp_idx);
    for chan_idx = [1,2,4]
        chanvec = squeeze(puncta_vector_filtered(:,exp_idx,chan_idx));
        
        %Remove entries larger than 3 stddevs above the mean
        mean_val = mean(chanvec);
        std_val = std(chanvec);
        chanvec(chanvec>mean_val+3*std_val) = mean_val+3*std_val;
        
        [values,binedges] = histcounts(chanvec,NUM_BUCKETS);
        
        
        %         b = bar(binedges(1:NUM_BUCKETS),values,colors{chan_idx});
        b = bar(1:NUM_BUCKETS,values,colors{chan_idx});
        b.FaceAlpha = 0.3;
        hold on;
        
    end
    title(sprintf('Experiment %i',exp_idx));
    xlabel('Histogram bucket indices');
    legend('Chan1','Chan2','Chan4');
end
hold off;

%% Assuming raw_pixels are loaded, let's try fitting two poissons
% raw pixels is of size = (params.NUM_ROUNDS,params.NUM_CHANNELS,2);

chan1_sig = raw_pixels{1,1,IDX_SIGNAL};
chan1_bg = raw_pixels{1,1,IDX_BACKGROUND};

%To remove outliers or not?
% percentiles_bg  = prctile(chan1_bg,[0,99]);
% percentiles_sig = prctile(chan1_sig,[0,99]);
% outlierIndex_bg = chan1_bg > percentiles_bg(2);
% chan1_bg(outlierIndex_bg) = [];
% outlierIndex_sig = chan1_sig > percentiles_sig(2);
% chan1_sig(outlierIndex_sig) = [];

min_val = min([chan1_sig; chan1_bg]);
chan1_sig = chan1_sig - min_val;
chan1_bg = chan1_bg - min_val;

mean_val = mean([chan1_sig; chan1_bg]);
chan1_sig = chan1_sig ./mean_val;
chan1_bg = chan1_bg ./mean_val;



%Realizing that we haven't been doing this on normalized data
figure;
subplot(2,1,1)

%Concatenate the two so we can get proper bucket
[values,binedges] = histcounts([chan1_bg; chan1_sig],NUM_BUCKETS);

%Then we have to scale all the data to range
[values_bg,binedges_bg] = histcounts(chan1_bg,binedges);
[values_sig,binedges_sig] = histcounts(chan1_sig,binedges);

b = bar(binedges(1:NUM_BUCKETS),values_bg/sum(values_bg),'b');
b.FaceAlpha = 0.3;
hold on;
b = bar(binedges(1:NUM_BUCKETS),values_sig/sum(values_sig),'r');
b.FaceAlpha = 0.3;
hold off;
title('Actual distribution');
legend('Background','Signal');

subplot(2,1,2)

lambda_sig = poissfit(chan1_sig);
lambda_bg = poissfit(chan1_bg);

test_vec = 1:max(binedges_sig); %the bigger numbever

plot(round(binedges(1:NUM_BUCKETS)), poisspdf(round(binedges(1:NUM_BUCKETS)),lambda_bg));
hold on;
plot(round(binedges(1:NUM_BUCKETS)), poisspdf(round(binedges(1:NUM_BUCKETS)),lambda_sig));
hold off;
legend('Background','Signal');
title('Fit distributions');


%% So if we pick a puncta and calculate the joint probability that all the 8
% pixels are in either one or the other, do we get a reasonable value?

central_puncta_indices= 5:6;
exp_idx = 1;
chan_idx = 1;
puncta_idx = 1;
subvolume = puncta_set(central_puncta_indices,...
    central_puncta_indices,...
    central_puncta_indices,...
    exp_idx,...
    chan_idx, puncta_idx);

pixels = subvolume(:);
prob_for_sig = poisspdf(pixels,lambda_sig);
prob_for_bg = poisspdf(pixels,lambda_bg);
score_for_signal = sum(log(prob_for_sig(prob_for_sig~=0)))
score_for_bg = sum(log(prob_for_bg(prob_for_bg~=0)))

%% generate 3d point cluster
colors = {'r.','g.','b.'};
transcripts_kmeansColor = zeros(size(transcripts));
for exp_idx = 1:3
    figure;
    experiment1_vector = squeeze(puncta_vector(:,exp_idx,[1,2,4]));
    [idx, C] = kmeans(experiment1_vector,3);
    hold on;
    
    for kidx = 1:3
        plot3(experiment1_vector(idx==kidx,1),...
            experiment1_vector(idx==kidx,2),...
            experiment1_vector(idx==kidx,3),...
            colors{kidx});
    end
    hold off;
    title(sprintf('Experiment %i comparing central pixels', exp_idx));
    grid on;
    
    transcripts_kmeansColor(:,exp_idx) = idx;
    transcripts_kmeansColor(transcripts_kmeansColor(:,exp_idx)==3,exp_idx) = 4;
end

%% For one quick exploration, use kmeans to generate clusters
%And use that score instead

% for exp_idx = 1:3
exp_idx = 1;
experiment_vector = squeeze(puncta_vector(:,exp_idx,[1,2,4]));


% end




