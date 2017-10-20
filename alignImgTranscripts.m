%Align to the known sequences
loadParameters
%,'transcripts','transcripts_confidence','pos_for_reference_round');
load(fullfile(params.punctaSubvolumeDir,'transcriptsv13_punctameannormed.mat'));

%pos = pos_for_reference_round;
%params.transcriptResultsDir = '/Users/Goody/Neuro/ExSeq/exseq20170524/6_transcripts';

%% Score it
clear perc_base
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

%% Get distribution of the ground truth
clear perc_base
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
MAXIMUM_MISSING_ROUNDS = 2;
transcript_ctr =1;
for p_idx = 1:size(transcripts,1)
    
    
    transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts(p_idx,4:end);
    
    %Sanity check: randomize the img_transcript
    
    %NEW: updating the round
    round_mask = (img_transcript ~= 0);
    
    if(sum(~round_mask)>MAXIMUM_MISSING_ROUNDS)
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
    if sum(scores==best_score)==1 %If there is a unique fit
        fit_idx = find(scores==best_score);
        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_confidence=transcripts_confidence(p_idx,4:end);
        transcript.known_sequence_matched = groundtruth_codes(fit_idx,:);
        transcript.distance_score= best_score;
        transcript.name = gtlabels{fit_idx};
        
        %The pos vector of centroids looks like this: pos(:,exp_idx,puncta_idx)
        %transcript.pos = squeeze(pos(:,:,p_idx));
    else
%         %If there is a tie between transcripts
%         [values, indices] = sort(scores,'ascend');
%         
%         %Get the first index that is great than the best score
%         %If this is 1, then there is a best fit
%         idx_last_tie = find(values>values(1),1)-1;
%         
%         %Find the bases that are different between the calls, choose the
%         %one that has a higher confidence for those differences
%         ground_truth_candidates = groundtruth_codes(indices(1:idx_last_tie),:);
% %         different_base_indices = (diff(ground_truth_candidates)~=0);
%         
%         metrics_difference = zeros(idx_last_tie,1);
%         for idx = 1:idx_last_tie
%             %Crop to just the transcript. TEMP
%             img_transcript_confidence = transcripts_confidence(p_idx,4:end);
%             metrics_difference(idx) = ...
%                 sum(img_transcript_confidence(img_transcript ~= groundtruth_codes(indices(idx),:))); 
%         end
%         %This whole thing is crappy, so just used as a temporary
%         %placeholder
%         
% %         [~,best_metric_index] = min(metrics_difference);
%         best_metric_index = randperm(length(metrics_difference),1);
        indices_that_tie_bestscore = find(scores==best_score);
        tiebreaker_placeholder = randperm(length(indices_that_tie_bestscore),1);
        
        best_gt_match = indices_that_tie_bestscore(tiebreaker_placeholder);
        
        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_confidence=transcripts_confidence(p_idx,4:end);
        transcript.known_sequence_matched = groundtruth_codes(best_gt_match,:);
        transcript.distance_score= best_score;
        
        transcript.name = gtlabels{best_gt_match};
    end
    
    transcript.pos = pos_for_reference_round(p_idx,:);
    
    transcript_objects{transcript_ctr} = transcript;
    hamming_scores(transcript_ctr) = best_score; %quick temporary output
    
    transcript_ctr = transcript_ctr+1;
    clear transcript; %Avoid any accidental overwriting
    
    if mod(p_idx,100) ==0
        fprintf('%i/%i matched\n',p_idx,size(puncta_set,6));
    end
    
end

hamming_scores
figure; histogram(hamming_scores);

filename_out = fullfile(params.transcriptResultsDir, 'transcript_objects.mat');
save(filename_out,'transcript_objects');

%% 
filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,1,'ch00'));
sample_img = load3DTif_uint16(filename_in);

%Make a four channel output image X,Y,Z of 3x3 puncta for hamming scores
outputImg = zeros([size(sample_img),max(hamming_scores)]);

%TEMP! This was left over in one experiment
padwidth = ceil(params.PUNCTA_SIZE/2);

for i = 1:length(transcript_objects)
    
    hamming_score = transcript_objects{i}.distance_score;
   
    centroid_pos = transcript_objects{i}.pos-padwidth;
    

    %Watch out for the XY shift
    y_indices = (centroid_pos(1) - 1):(centroid_pos(1) + 1);
    y_indices(y_indices<1)=[];y_indices(y_indices>size(sample_img,1))=[];
    
    x_indices = (centroid_pos(2) - 1):(centroid_pos(2) + 1);
    x_indices(x_indices<1)=[];x_indices(x_indices>size(sample_img,2))=[];
    
    z_indices = (centroid_pos(3) - 1):(centroid_pos(3) + 1);
    z_indices(z_indices<1)=[];z_indices(z_indices>size(sample_img,3))=[];
    
    outputImg(y_indices,x_indices,z_indices,hamming_score+1) = 100;

end

for hamming_range = 1:size(outputImg,4)
    output_filename = fullfile(params.punctaSubvolumeDir,sprintf('punctaScores_hamming=%i.tif',hamming_range-1));
    save3DTif_uint16(squeeze(outputImg(:,:,:,hamming_range)),output_filename);
end
% %%
% figure;
% imagesc(max(outputImg,[],3))
% 
% figure; 
% imagesc(transcript_img)
%%
hamming_scores = zeros(20,1);
for p = 1:length(transcript_objects)
    hamming_scores(transcript_objects{p}.distance_score+1) = hamming_scores(transcript_objects{p}.distance_score+1)+1;
end
figure; bar(hamming_scores)