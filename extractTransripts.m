% Quick demo script towards base calling

%% Load the data
%Load the puncta subvolumes from the baseCalling.m (todo: rename)
dir_input = 'rajlab/splintr1/';
load(fullfile(dir_input,'rois_votednonnormed.mat'))

%Note: puncta_set is only the set of puncta locations that have full 
%subvolumes

%% Loop through, producing visual "base calls" per puncta 

NUM_ROUNDS = 3;
NUM_COLORS = 4;
COLOR_VEC = [1,2,4]; %Which channels are we comparing? (in case of empty chan)
DISTANCE_FROM_CENTER = 2.5;

puncta_vector = zeros(size(puncta_set,6),NUM_ROUNDS,NUM_COLORS);
for z = 1:size(puncta_vector,1)
    for exp_idx = 1:NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,z));
        
        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        
        [max_chan, confidence,scores] = chooseChannelMod(punctaset_perround,COLOR_VEC,DISTANCE_FROM_CENTER);
        puncta_vector(z,exp_idx,:) = scores;
    end
    
    if(mod(z,100)==0)
        fprintf('Processing puncta #%i out of %i \n',z, size(puncta_vector,1));
    end
    
end

%%

% Rows are puncta, columns are rounds


transcripts = zeros(length(good_puncta_indices),NUM_ROUNDS);
transcripts_confidence = zeros(length(good_puncta_indices),NUM_ROUNDS);
puncta_ctr = 1;

for puncta_idx = good_puncta_indices
    

    
    answer_vector = zeros(NUM_ROUNDS,1);
    confidence_vector = zeros(NUM_ROUNDS,1);
    for exp_idx = 1:NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));

        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));

       [max_chan, confidence] = chooseChannel(punctaset_perround,[1,2,4],DISTANCE_FROM_CENTER); 
       answer_vector(exp_idx) = max_chan;
       confidence_vector(exp_idx) = confidence;
    end
    
    transcripts(puncta_ctr,:) = answer_vector;
    transcripts_confidence(puncta_ctr,:) = confidence_vector;
    pos(puncta_ctr,:) = [Y(puncta_idx),X(puncta_idx),Z(puncta_idx)];
    puncta_ctr = puncta_ctr +1;
    
    fprintf('Showing puncta #%i out of %i \n',puncta_ctr, length(good_puncta_indices));
    

end

save(fullfile(dir_input,'transcriptsv3_nonnormed.mat'),'transcripts','transcripts_confidence','pos');