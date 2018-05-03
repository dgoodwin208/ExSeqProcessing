%New base-calling, correcting for previous rounds fluorescence

%load('/mp/nas0/ExSeq/AutoSeqHippocampus/b3/6_transcripts/exseqauto-b3_transcriptmatches_objectsFULL.mat');
load(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)));
load('groundtruth_dictionary.mat');
%% First remove any empty rounds
discarded_puncta = find(cellfun(@isempty,transcript_objects));
transcript_objects(discarded_puncta) = [];

allIntensitiesPerTranscriptCell = cellfun(@(x) [squeeze(x.img_transcript_absValuePixel)], transcript_objects,'UniformOutput',0);

puncta_indices_with_empty_vols = [];
ctr = 1;
for idx = 1:length(allIntensitiesPerTranscriptCell)
   intensities =  allIntensitiesPerTranscriptCell{idx};
   
   if any(sum(intensities>0,2)==0)
       puncta_indices_with_empty_vols(ctr)= idx;
       ctr = ctr+1;
   end
end
fprintf('Removing %i transcripts that are missing a round\n',length(puncta_indices_with_empty_vols));
transcript_objects(puncta_indices_with_empty_vols) = [];

allIntensitiesPerTranscriptCell = cellfun(@(x) [squeeze(x.img_transcript_absValuePixel)], transcript_objects,'UniformOutput',0);
size(allIntensitiesPerTranscriptCell{1});
allIntensitiesPerTranscript = zeros(length(allIntensitiesPerTranscriptCell),size(allIntensitiesPerTranscriptCell{1},1),size(allIntensitiesPerTranscriptCell{1},2));
for p = 1:length(allIntensitiesPerTranscriptCell)
    allIntensitiesPerTranscript(p,:,:) = allIntensitiesPerTranscriptCell{p};
end

%If we want to do a hybrid of the naive and slightly advanced base call
%needsToBeReCalled = cell2mat(cellfun(@(x) [x.hamming_score>1], transcript_objects,'UniformOutput',0));
%Or we just re-call eveerything:
needsToBeReCalled = ones(length(transcript_objects),1);
%First we base-call the two ligation round

transcripts_new = zeros(size(allIntensitiesPerTranscript,1),20);

for rnd_idx = 1:10
    
    %Convert all the intensities into Z-scores
    puncta_intensities = squeeze(allIntensitiesPerTranscript(:,rnd_idx,:));
    
    puncta_zscore = zscore(puncta_intensities);
    
    [~,calls] = max(puncta_zscore,[],2);
    transcripts_new(:,rnd_idx) = calls;
end

for rnd_idx = 11:20
    
    
    
    for c_idx = 1:4
        %Get the indices of the of channel that agree
        puncta_matching_previous_base = find(transcripts_new(:,rnd_idx-5)==c_idx);
        puncta_NOTmatching_previous_base = find(transcripts_new(:,rnd_idx-5)~=c_idx);
        
        %Get the statistics of all the change in intensity based on the
        %previous round
        color_match_intensity = allIntensitiesPerTranscript(puncta_matching_previous_base,rnd_idx,c_idx);
        color_NOTmatch_intensity = allIntensitiesPerTranscript(puncta_NOTmatching_previous_base,rnd_idx,c_idx);
        colorMATCH_mean = mean(color_match_intensity);
        colorNOTMATCH_means = mean(color_NOTmatch_intensity);

        %Get a color correctino factor:
        color_correction_factor = colorMATCH_mean - colorNOTMATCH_means;
        
        %apply the color correction factor:
        
        allIntensitiesPerTranscript(puncta_matching_previous_base,rnd_idx,c_idx) = ...
            allIntensitiesPerTranscript(puncta_matching_previous_base,rnd_idx,c_idx) ...
            - color_correction_factor;
        
    end
    
    puncta_intensities = squeeze(allIntensitiesPerTranscript(:,rnd_idx,:));
    
    puncta_zscore = zscore(puncta_intensities);
    
    [~,calls] = max(puncta_zscore,[],2);
    transcripts_new(:,rnd_idx) = calls;
    
end

%% Create a new transcript object

%params.transcriptResultsDir = '';
if ~exist('gtlabels','var')
    loadGroundTruth;
end
%

transcript_objects_new = cell(size(transcripts_new,1),1);

output_cell = {}; ctr = 1;

tic
for p_idx = 1:size(transcripts_new,1)

     
    transcript_old = transcript_objects{p_idx};
    
    %If the original base caller did a good enough job
    if ~needsToBeReCalled(p_idx)
        transcript_old.reprocessed= false;
        transcript_objects_new{p_idx} = transcript_old;
        continue;
    end

    transcript = struct;
    
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts_new(p_idx,4:end);
   
    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes(:,1:end)==img_transcript);

    %Calculate the hamming distance (now subtracting primer length)
    scores = length(img_transcript)- sum(hits,2);
    [values, indices] = sort(scores,'ascend');

    best_score = values(1);
    %Get the first index that is great than the best score
    %If this is 1, then there is a unique fit
    
    %Rewriting a line of code to hopefully make this step faster
%     idx_last_tie = find(values>best_score,1)-1;
    idx_last_tie = 1;
    while values(idx_last_tie)<=best_score 
        idx_last_tie = idx_last_tie +1; 
    end
    idx_last_tie = idx_last_tie-1;
    
    %Assuming the groundtruth options are de-duplicated
    %Is there a perfect match to the (unique ) best-fit
    transcript.img_transcript=transcripts_new(p_idx,:);
    transcript.img_transcript_absValuePixel=transcript_old.img_transcript_absValuePixel;
    
    transcript.pos = transcript_old.pos; 
    transcript.hamming_score = best_score;
    transcript.numMatches = idx_last_tie;
    transcript.nn_distance = transcript_old.nn_distance;
    
    if best_score <=1
        row_string = sprintf('%i,%s,',p_idx,mat2str(img_transcript));
        
        if idx_last_tie==1 
            transcript.known_sequence_matched = groundtruth_codes(indices(idx_last_tie),:);
            transcript.name = gtlabels{indices(idx_last_tie)};        
        end
        
        for tiedindex = 1:idx_last_tie
            row_string = strcat(row_string,sprintf('%s,',gtlabels{indices(tiedindex)}));
        end
        row_string = sprintf('%s\n',row_string);
        fprintf('%s',row_string);
        output_cell{ctr} = row_string; ctr = ctr+1;
    end
    
    transcript.reprocessed= true;
    transcript_objects_new{p_idx} = transcript;

    clear transcript; %Avoid any accidental overwriting

    if mod(p_idx,500) ==0
       toc; tic 
       fprintf('%i/%i matched\n',p_idx,size(transcripts_new,6));
    end
end
fprintf('Done!\n');

output_csv = strjoin(output_cell,'');
params.FILE_BASENAME = 'exseqauto-b2';
output_file = fullfile(params.transcriptResultsDir,sprintf('%s_matchednewcodes.csv',params.FILE_BASENAME));

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

%%
save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objectsMODIFIED.mat',params.FILE_BASENAME)),'transcript_objects_new','transcript_objects','-v7.3');
fprintf('Saved transcript_matches_objects!\n');
