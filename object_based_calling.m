loadParameters;
load(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedroisv12.mat',params.FILE_BASENAME)));

%%



transcripts = zeros(size(puncta_set,6),params.NUM_ROUNDS);
transcripts_confidence = zeros(size(puncta_set,6),params.NUM_ROUNDS);
transcripts_sizes = zeros(size(puncta_set,6),params.NUM_ROUNDS);
for p_idx = 1:size(puncta_set,6)
    puncta = puncta_set(:,:,:,:,:,p_idx);

    for rnd_idx = 1:params.NUM_ROUNDS
        puncta_exp = squeeze(puncta(:,:,:,rnd_idx,:));
        [scores,num_pixels_in_call] = callRoundROI(puncta_exp);
        [vals_sorted,chans_sorted] = sort(scores,'descend');
        transcripts(p_idx,rnd_idx) = chans_sorted(1);
        transcripts_confidence(p_idx,rnd_idx) = vals_sorted(1)/vals_sorted(2);
        transcripts_sizes(p_idx,rnd_idx) = num_pixels_in_call;
    end
    
    if mod(p_idx,1000) ==0
        fprintf('%i/%i called\n',p_idx,size(puncta_set,6));
    end
   
    
end

%% Score it
for base_idx = 1:params.NUM_CHANNELS
    perc_base(:,base_idx) = sum(transcripts==base_idx,1)/size(transcripts,1);
end
figure;
plot(perc_base*100,'LineWidth',2)
legend('Chan1 - FITC','Chan2 - CY3', 'Chan3 - Texas Red', 'Chan4 - Cy5');
title(sprintf('Percentage of each base across rounds for %i puncta',size(transcripts,1)));


%% Load Ground truth information

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

%% Make sets of transcripts and create a new transcript object

%AS A TEMPORARY HACK, only process the uniques
%In the long run this will be helpful too when implemented fully
%[transcripts, ia, ic] = unique(transcripts,'rows');

transcript_objects = cell(size(transcripts,1),1);

for p_idx = 1:size(transcripts,1)

    transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts(p_idx,4:end);
    %Sanity check: randomize the img_transcript

    %img_transcript = img_transcript(randperm(length(img_transcript)));


    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes==img_transcript);


    %Calculate the hamming distance (now subtracting primer length)
    scores = length(img_transcript)- sum(hits,2);
    [values, indices] = sort(scores,'ascend');

    %Get the first index that is great than the best score
    %If this is 1, then there is a best fit
    idx_last_tie = find(values>values(1),1)-1;

    %Assuming the groundtruth options are de-duplicated
    %Is there a perfect match to the (unique ) best-fit
    if idx_last_tie==1
        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_confidence=transcripts_confidence(p_idx,:);
        transcript.img_transcripts_size=transcripts_sizes(p_idx,:);
        transcript.known_sequence_matched = groundtruth_codes(indices(idx_last_tie),:);
        transcript.distance_score= values(idx_last_tie);
        transcript.name = gtlabels{indices(idx_last_tie)};
        transcript.prob_error= -1; %Saying N/A;

        transcript.pos = pos(p_idx,:); %TODO get position included

    else

        %Difference metric: sum of probabilities,that the original call
        %was wrong. This is calcuated per round that has a disagreement
        %between the groundtruth option and the image sequence
        metrics_difference = zeros(idx_last_tie,1);
        for idx = 1:idx_last_tie
            metrics_difference = ...
                -1*sum(transcripts_confidence(...
                img_transcript== groundtruth_codes(indices(idx),:)...
                ,:));
        end

        [val_best_second_place,idx_best_second_place]=max(metrics_difference);

        transcript.img_transcript=transcripts(p_idx,:);
   %     transcript.img_transcript_votes=transcripts_votes(p_idx,:);
    %    transcript.img_transcript_calls=transcripts_calls(:,:,p_idx);
        
        transcript.known_sequence_matched = groundtruth_codes(indices(idx_best_second_place),:);
        transcript.distance_score= values(idx_last_tie);
        transcript.prob_error= val_best_second_place;
        transcript.pos = pos(p_idx,:); %TODO get position included


        transcript.name = gtlabels{indices(idx_best_second_place)};
    end

    transcript_objects{p_idx} = transcript;

    clear transcript; %Avoid any accidental overwriting

    if mod(p_idx,1000) ==0
        fprintf('%i/%i matched\n',p_idx,size(puncta_set,6));
    end
end


%save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)),'transcript_objects','-v7.3');
                
%%
                
 %               error_reads = zeros(17,1);
 %               indices_for_specific_distance = []; ctr = 1;
 %               for idx = 1:size(transcript_objects,1)
 %                   if transcript_objects{idx}.distance_score-3==3
 %                       indices_for_specific_distance(ctr) = idx;
 %                       ctr = ctr +1;
 %                   end
 %                   addition = (transcript_objects{idx}.img_transcript(4:end)~=transcript_objects{idx}.known_sequence_matched);
 %                   error_reads = error_reads + double(addition');
 %               end
                
%                puncta_set_cropped =  puncta_set(:,:,:,:,:,indices_for_specific_distance);
%                 save(fullfile(params.transcriptResultsDir,sprintf('%s_RANDpunctanormed_crop3.mat',params.FILE_BASENAME)),'puncta_set_cropped','-v7.3');             img_transcript== groundtruth_codes(indices(idx),:)...
%                     ,:));
%             end
%             
%             [val_best_second_place,idx_best_second_place]=max(metrics_difference);
%             
%             transcript.img_transcript=transcripts(p_idx,:);
%             transcript.img_transcript_votes=transcripts_votes(p_idx,:);
%             transcript.img_transcript_calls=transcripts_calls(:,:,p_idx);
%             transcript.known_sequence_matched = groundtruth_codes(indices(idx_best_second_place),:);
%             transcript.distance_score= values(idx_last_tie);
%             transcript.prob_error= val_best_second_place;
%             transcript.pos = pos(p_idx,:); %TODO get position included
%             
%             
%             transcript.name = gtlabels{indices(idx_best_second_place)};
%         end
%         
%         transcript_objects{p_idx} = transcript;
%         
%         clear transcript; %Avoid any accidental overwriting
%         
%         if mod(p_idx,1000) ==0
%             fprintf('%i/%i matched\n',p_idx,size(puncta_set,6));
%         end
%     end
%     
%     
     save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)),'transcript_objects','-v7.3');
%     
    
