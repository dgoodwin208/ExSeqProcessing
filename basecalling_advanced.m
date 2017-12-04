%New base-calling, correcting for previous rounds fluorescence


allIntensitiesPerTranscriptCell = cellfun(@(x) [squeeze(x.img_transcript_absValuePixel)], transcript_objects,'UniformOutput',0);
size(allIntensitiesPerTranscriptCell{1});
allIntensitiesPerTranscript = zeros(length(allIntensitiesPerTranscriptCell),size(allIntensitiesPerTranscriptCell{1},1),size(allIntensitiesPerTranscriptCell{1},2));
for p = 1:length(allIntensitiesPerTranscriptCell)
    allIntensitiesPerTranscript(p,:,:) = allIntensitiesPerTranscriptCell{p};
end


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

hamming_scores = zeros(size(transcripts_new,1),1);
for p_idx = 1:size(transcripts_new,1)

    
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts_new(p_idx,4:end);
    
    %Sanity check: randomize the img_transcript
%     img_transcript = img_transcript(randperm(length(img_transcript)));

    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes(:,1:end)==img_transcript);

    %Calculate the hamming distance (now subtracting primer length)
    scores = length(img_transcript)- sum(hits,2);
    [values, indices] = sort(scores,'ascend');

    best_score = values(1);
    hamming_scores(p_idx) = best_score;
    
    if mod(p_idx,100)==0
       fprintf('%i/%i processed!\n',p_idx,size(transcripts_new,1));
    end
end
