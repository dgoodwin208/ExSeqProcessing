

load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_finalmatches.mat',params.FILE_BASENAME)));


load('groundtruth_dictionary.mat')
%% Convert all the data into zscores (very cheap base calling)
puncta_set_normed = zeros(size(puncta_set));
clear chan_col; %Just in case, otherwise the for loop can error.
for c = params.COLOR_VEC
    chan_col(:,c) = reshape(puncta_set(:,:,:,:,c,:),[],1);
end

% cols_normed = quantilenorm(chan_col);
cols_normed = zscore(chan_col);

for c = params.COLOR_VEC
    puncta_set_normed(:,:,:,:,c,:) = reshape(cols_normed(:,c),size(squeeze(puncta_set(:,:,:,:,c,:))));
end


%% Compare the top 10 brightest (zscore) pixels in each puncta across channels

path_indices = 1:size(final_positions,1);

%Pre-initialize the cell arrray and determine the basecalls
chans = zeros(params.PUNCTA_SIZE^3,4);
base_calls_quickzscore = zeros(length(path_indices),params.NUM_ROUNDS);
base_calls_quickzscore_confidence = zeros(length(path_indices),params.NUM_ROUNDS);
for p_idx= 1:length(path_indices) 
    
    path_idx = path_indices(p_idx);
    
    for rnd_idx = 1:params.NUM_ROUNDS
        
        %Load and vectorize the puncta_subset
        for c = 1:params.NUM_CHANNELS
            chantemp = puncta_set_normed(:,:,:,rnd_idx,c,p_idx);
            chans(:,c) = chantemp(:)';
        end
        
        sorted_chans = sort(chans,1,'descend');
        %Take the mean of the top 10 values
        scores = mean(sorted_chans(1:10,:),1);
        %add the minimum score to shift everything non-zero
        scores = scores - min(scores);
        
        %and the new baseguess 
        [~, newbaseguess] = max(scores);
        base_calls_quickzscore(p_idx,rnd_idx) = newbaseguess;
        
        [scores_sorted,~] = sort(scores,'descend');
        base_calls_quickzscore_confidence(p_idx,rnd_idx) = scores_sorted(1)/(scores_sorted(1)+ scores_sorted(2));
    end
end

[unique_transcipts,~,~] = unique(base_calls_quickzscore,'rows');
fprintf('Found %i transcripts, %i of which are unique\n',size(base_calls_quickzscore,1),size(unique_transcipts,1));

%% For the final puncta, score each index with it's distance to nearests neightbor
% filename_centroidsMOD = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels_demerged.mat',params.FILE_BASENAME));
% load(filename_centroidsMOD);

%For each puncta, find it's nearest neighbor in the same round
[IDX,D] = knnsearch(final_positions,final_positions,'K',2); %getting four other options

spacings = zeros(size(centroids,1),1);
%For each puncta, ignore the mapping to itself, and note the number of
%possible merge mistakes for this puncta
for puncta_idx = 1:size(centroids,1)
    %When doing KNN with itself, D(1) will be 0
    spacings(puncta_idx) = floor(D(puncta_idx,2));
end

%% Make sets of transcripts and create a new transcript object
if ~exist('gtlabels','var')
    loadGroundTruth;
end
%
transcript_objects = cell(size(base_calls_quickzscore,1),1);

output_cell = {}; ctr = 1;

for p_idx = 1:size(base_calls_quickzscore,1)

    transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = base_calls_quickzscore(p_idx,4:end);
    
    %Sanity check: randomize the img_transcript
    %img_transcript = img_transcript(randperm(length(img_transcript)));

    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes==img_transcript);

    %Calculate the hamming distance (now subtracting primer length)
    scores = length(img_transcript)- sum(hits,2);
    [values, indices] = sort(scores,'ascend');

    best_score = values(1);
    %Get the first index that is great than the best score
    %If this is 1, then there is a best fit
    idx_last_tie = find(values>best_score,1)-1;

    %Assuming the groundtruth options are de-duplicated
    %Is there a perfect match to the (unique ) best-fit
    transcript.img_transcript=base_calls_quickzscore(p_idx,:);
    transcript.img_transcript_confidence=base_calls_quickzscore_confidence(p_idx,:);
    transcript.pos = final_positions(p_idx,:); 
    transcript.hamming_score = best_score;
    transcript.nn_distance = spacings(p_idx);
    
    if best_score <=1
        row_string = sprintf('%i,%s,',p_idx,mat2str(img_transcript));
        
        if idx_last_tie==1 
            transcript.known_sequence_matched = groundtruth_codes(indices(idx_last_tie),:);
            transcript.name = gtlabels{indices(idx_last_tie)};        
%         else
%             fprintf('Found a non-unique match under hammign 1\n');
        end
        
        for tiedindex = 1:idx_last_tie
        row_string = strcat(row_string,sprintf('%s,',gtlabels{indices(tiedindex)}));
        end
        row_string = sprintf('%s\n',row_string);
        fprintf('%s',row_string);
        output_cell{ctr} = row_string; ctr = ctr+1;
    end
    
    transcript_objects{p_idx} = transcript;

    clear transcript; %Avoid any accidental overwriting

    if mod(p_idx,100) ==0
        fprintf('%i/%i matched\n',p_idx,size(puncta_set,6));
    end
end
fprintf('Done!\n');

output_csv = strjoin(output_cell,'');

output_file = fullfile(params.transcriptResultsDir,sprintf('%s_groundtruthcodes.csv',params.FILE_BASENAME));

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

%%
save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)),'transcript_objects','-v7.3');
fprintf('Saved trnscript_matches_objects!\n');