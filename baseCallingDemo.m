% Quick demo script towards base calling

%% Load the data
%Load a bunch of parameters left 
% load('cropped_roi_parameters.mat');
%if you want to load non-normalized, just change this line
% load('puncta_set_normalized.mat'); 

load('rajlab/ExSeqCulture/roi_parameters_and_punctaset.mat')
good_puncta_indices = setdiff(1:num_puncta,bad_puncta_indices);
%% Loop through, producing visual "base calls" per puncta 

% Rows are sequencing rounds, columns are channels, press enter to go to
% next one

transcripts = zeros(length(good_puncta_indices),NUM_ROUNDS);
transcripts_confidence = zeros(length(good_puncta_indices),NUM_ROUNDS);
puncta_ctr = 1;

DISTANCE_FROM_CENTER = 2.5;

% figure(1); %DEBUG
% subplot(NUM_ROUNDS,NUM_CHANNELS,1); %DEBUG

for puncta_idx = good_puncta_indices
    
    %     subplot_idx = 1; %DEBUG
    
    answer_vector = zeros(NUM_ROUNDS,1);
    confidence_vector = zeros(NUM_ROUNDS,1);
    for exp_idx = 1:NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));

        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
%         for c_idx = 1:NUM_CHANNELS
% 
%             clims = [min_intensity,max_intensity];
% %             subplot(NUM_ROUNDS,NUM_CHANNELS,subplot_idx); %DEBUG
%             data = squeeze(punctaset_perround(:,:,:,c_idx));
%             
%             img = max(data,[],3);
%             [xymax,imax,xymin,imin] = extrema2(img);
%             [x,y] = ind2sub(size(img),imax);
%             
% %             imagesc(img,clims); hold on;  %DEBUG
% %             plot(y,x,'r+'); hold off;   %DEBUG
% %             axis off; colormap gray;    %DEBUG
% %             subplot_idx = subplot_idx+1; %DEBUG
%         end
       [max_chan, confidence] = chooseChannel(punctaset_perround,4,DISTANCE_FROM_CENTER); 
       answer_vector(exp_idx) = max_chan;
       confidence_vector(exp_idx) = confidence;
    end
    
    transcripts(puncta_ctr,:) = answer_vector;
    transcripts_confidence(puncta_ctr,:) = confidence_vector;
    
    puncta_ctr = puncta_ctr +1;
    
    fprintf('Showing puncta #%i out of %i. press Enter to continue \n',puncta_idx, length(good_puncta_indices));
    
%     [answer_vector,confidence_vector] %DEBUG
%     pause %DEBUG
end

save('transcripts_center-only.mat','transcripts','transcripts_confidence');