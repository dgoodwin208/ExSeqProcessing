%Assuming puncta_set is loaded and filtered by 

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
