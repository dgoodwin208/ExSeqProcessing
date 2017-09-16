%Align to the known sequences

%,'transcripts','transcripts_confidence','pos_for_reference_round');
load(fullfile(params.punctaSubvolumeDir,'transcriptsv13_punctameannormed.mat'));

params.transcriptResultsDir = '/Users/Goody/Neuro/ExSeq/exseq20170524/6_transcripts';

%% Score it
for base_idx = 1:params.NUM_CHANNELS
    perc_base(:,base_idx) = sum(transcripts==base_idx,1)/size(transcripts,1);
end
figure;
% Chan 1 = Blue
% Chan 2 = Green
% Chan 3 = Magenta
% Chan 4 = Red

plot(perc_base(:,1)*100,'b','LineWidth',2); hold on;
plot(perc_base(:,2)*100,'g','LineWidth',2)
plot(perc_base(:,3)*100,'m','LineWidth',2)
plot(perc_base(:,4)*100,'r','LineWidth',2); hold off;
legend('Chan1 - FITC','Chan2 - CY3', 'Chan3 - Texas Red', 'Chan4 - Cy5');
title(sprintf('Percentage of each base across rounds for %i puncta',size(transcripts,1)));


%% Load Ground truth information
%load(fullfile(params.transcriptResultsDir,'groundtruth_codes.mat'));
%Turn the Nx3 array into a set of strings
%gtlabels = {};
%for i = 1:size(groundtruth_codes,1)
%    transcript_string = '';
%    for c = 1:size(groundtruth_codes,2)
%        transcript_string(c) = num2str(groundtruth_codes(i,c));
%    end
%    gtlabels{i}=transcript_string;
%end
t = fastaread(fullfile(params.transcriptResultsDir, 'SOLiD_like_reads_in_Illumina_unique_17.fa'));

groundtruth_codes = zeros(length(t),17);
gtlabels = cell(length(t),1);
for idx = 1:length(t)
    seq = t(idx).Sequence;
    
    for c = 1:length(seq)
        if seq(c)=='A'
            groundtruth_codes(idx,c)=1;
        elseif seq(c)=='C'
            groundtruth_codes(idx,c)=2;
        elseif seq(c)=='G'
            groundtruth_codes(idx,c)=3;
        elseif seq(c)=='T'
            groundtruth_codes(idx,c)=4;
        end
        
    end
    gtlabels{idx}=t(idx).Header;
end

%% Get distributio
for base_idx = 1:params.NUM_CHANNELS
    perc_base(:,base_idx) = sum(groundtruth_codes==base_idx,1)/size(groundtruth_codes,1);
end
figure;
% Chan 1 = Blue
% Chan 2 = Green
% Chan 3 = Magenta
% Chan 4 = Red

plot(perc_base(:,1)*100,'b','LineWidth',2); hold on;
plot(perc_base(:,2)*100,'g','LineWidth',2)
plot(perc_base(:,3)*100,'m','LineWidth',2)
plot(perc_base(:,4)*100,'r','LineWidth',2); hold off;
legend('A - Chan1 - FITC','C - Chan2 - CY3', 'G - Chan3 - Texas Red', 'T - Chan4 - Cy5');
title(sprintf('Percentage of each base of ground truth, size= %i reads',size(groundtruth_codes,1)));

%%
% hamming_scores = zeros(size(transcripts,1),1);
% par_factor = 5;

transcript_ctr =1;
for p_idx = 1:size(transcripts,1)
    
    
    transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts(p_idx,4:end);
    
    %Sanity check: randomize the img_transcript
    
    %NEW: updating the round
    round_mask = (img_transcript ~= 0);
    
    if(sum(~round_mask)>2)
        fprintf('index %i has %i/%i zeros\n',p_idx,sum(~round_mask),length(img_transcript));
        continue;
    end
    
    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes(:,round_mask)==img_transcript(round_mask));
    
    %Calculate the hamming distance
    scores = length(img_transcript)- sum(hits,2) - sum(~round_mask);
    
    best_score = min(scores);
    
    %Assuming the groundtruth options are de-duplicated
    %Is there a perfect match to the (unique ) best-fit
    if sum(scores==best_score)==1 %If there is a perfect fit
        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_confidence=transcripts_confidence(p_idx,4:end);
        transcript.known_sequence_matched = groundtruth_codes(indices(idx_last_tie),:);
        transcript.distance_score= values(idx_last_tie);
        transcript.name = gtlabels{indices(idx_last_tie)};
        
        %The pos vector of centroids looks like this: pos(:,exp_idx,puncta_idx)
        transcript.pos = squeeze(pos(:,:,p_idx));
        
    else
        %If there is a tie between transcripts
        [values, indices] = sort(scores,'ascend');
        
        %Get the first index that is great than the best score
        %If this is 1, then there is a best fit
        idx_last_tie = find(values>values(1),1)-1;
        
        %Find the bases that are different between the calls, choose the
        %one that has a higher confidence for those differences
        ground_truth_candidates = groundtruth_codes(indices(1:idx_last_tie),:);
        different_base_indices = (diff(ground_truth_candidates)~=0);
        
        metrics_difference = zeros(idx_last_tie,1);
        for idx = 1:idx_last_tie
            %Crop to just the transcript. TEMP
            img_transcript_confidence = transcripts_confidence(p_idx,4:end);
            metrics_difference(idx) = ...
                sum(img_transcript_confidence(img_transcript ~= groundtruth_codes(indices(idx),:))); 
        end
        %This whole thing is crappy, so just used as a temporary
        %placeholder
        
        [~,best_metric_index] = min(metrics_difference);
        
        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_confidence=transcripts_confidence(p_idx,4:end);
        transcript.known_sequence_matched = groundtruth_codes(indices(best_metric_index),:);
        transcript.distance_score= values(idx_last_tie);
        
        %The pos vector of centroids looks like this: pos(:,exp_idx,puncta_idx)
        transcript.pos = squeeze(pos(:,:,p_idx));
        
        transcript.name = gtlabels{indices(best_metric_index)};
    end
    
    transcript_objects{transcript_ctr} = transcript;
    hamming_scores(transcript_ctr) = min(scores); %quick temporary output
    
    transcript_ctr = transcript_ctr+1;
    clear transcript; %Avoid any accidental overwriting
    
    if mod(p_idx,1000) ==0
        fprintf('%i/%i matched\n',p_idx,size(puncta_set,6));
    end
    
end

%% 
outputImg = zeros(size(experiment_set_masked));

skipped_puncta_ctr = 1;
img_ctr = 1;
for i = 1:length(transcript_objects)
    
    hamming_score = transcript_objects{i}.distance_score;
    if hamming_score>1
        continue;
    end
    
   
    centroid = transcript_objects{i}.pos(:,3);
    transcript_img(img_ctr,:) = transcript_objects{i}.known_sequence_matched;
    img_ctr = img_ctr+1;
    try
    x_indices = centroid(1) - 1:centroid(1) + 1;
    y_indices = centroid(2) - 1:centroid(2) + 1;
    z_indices = centroid(3) - 1:centroid(3) + 1;
    
    outputImg(y_indices,x_indices,z_indices) = hamming_score*20;
    catch
        fprintf('Puncta %i has zeros in round 6\n',i);
        skipped_puncta_ctr   = skipped_puncta_ctr+1;
    end
end
skipped_puncta_ctr

save3DTif_uint16(outputImg,'punctaColorsByHammingScore.tif');

%%
figure;
imagesc(max(outputImg,[],3))

figure; 
imagesc(transcript_img)