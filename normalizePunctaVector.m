%Load the ROIS, calculate normalized intensities, then make transcript base
%calls

loadParameters;
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));

%The puncta_set vector is of dimensions:
%(PUNCTA_SIZE,PUNCTA_SIZE,PUNCTA_SIZE,NUM_ROUNDS,NUM_CHANNELS,num_puncta)

%Let's get all the pixels per channel

%% Vectorize the entire set of puncta into one column per channel

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
      
    %How many unique values do we see per channel? (only useful to check
    %for unintentional 8-bit data)
    figure('Visible','off');
    subplot(length(params.COLOR_VEC),1,1)
    for c = params.COLOR_VEC
        subplot(length(params.COLOR_VEC),1,c)
        h = histogram(chan_col(:,c),100);
        unique_vals_count = length(unique(chan_col(:,c)));
        title(sprintf('Exp %i: Histogram of Channel %i. Unique values %i',exp_idx,c, unique_vals_count));
    end
       

    chan_col = chan_col - repmat(min(chan_col,[],1),size(chan_col,1),1);
    mean_vec = mean(chan_col,1);
    cols_normed = chan_col ./ repmat(mean_vec,size(chan_col,1),1);

    for c = params.COLOR_VEC
        puncta_set_normed(:,:,:,exp_idx,c,:) = reshape(cols_normed(:,c),squeeze(size(puncta_set(:,:,:,exp_idx,c,:))));
    end

    fprintf('Normalized the puncta for sequencing round %i\n',exp_idx);
end


% Rows are sequencing rounds, columns are channels, press enter to go to
% next one
transcripts = zeros(size(puncta_set_normed,6),params.NUM_ROUNDS);
transcripts_confidence = zeros(size(puncta_set_normed,6),params.NUM_ROUNDS);
pos = zeros(size(puncta_set_normed,6),3);

for puncta_idx = 1:size(puncta_set_normed,6)
    
    answer_vector = zeros(params.NUM_ROUNDS,1);
    confidence_vector = zeros(params.NUM_ROUNDS,1);
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set_normed(:,:,:,exp_idx,:,puncta_idx));
        
        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        
        [max_chan, confidence] = chooseChannel(punctaset_perround,params.COLOR_VEC,params.DISTANCE_FROM_CENTER);
        answer_vector(exp_idx) = max_chan;
        confidence_vector(exp_idx) = confidence;
    end
    
    transcripts(puncta_idx,:) = answer_vector;
    transcripts_confidence(puncta_idx,:) = confidence_vector;
    pos(puncta_idx,:) = [Y(puncta_idx),X(puncta_idx),Z(puncta_idx)];
    
    fprintf('Calling base puncta #%i out of %i \n',puncta_idx, size(puncta_set_normed,6));
end

save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptsv9.mat',params.FILE_BASENAME)),'transcripts','transcripts_confidence','pos');
save(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedrois.mat',params.FILE_BASENAME)),'puncta_set_normed');
disp('Completed normalizePuncta.m and saved the transcriptsv9 mat file in the transcripts folder');
