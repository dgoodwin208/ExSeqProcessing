%% Exploring NMF on the splintr data
loadParameters;
% params.NUM_ROUNDS=4;

% params.REFERENCE_ROUND_PUNCTA =5;
% load('/Users/Goody/Neuro/ExSeq/rajlab/splintr2roi1_mackerel_2ndpass/splintrseqroi1_puncta_rois.mat');
%Used to be normedoisv12 - DG 08262017
load(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedroisv12.mat',params.FILE_BASENAME)));

fprintf('Size of puncta: %i\n',size(puncta_set,1));
%%
PSIZE = params.PUNCTA_SIZE;
% p_idx = 500;
% puncta = puncta_set(:,:,:,:,:,p_idx);
center_pixels = 5:6;
transcripts = zeros(size(puncta_set,6),params.NUM_ROUNDS);
transcripts_votes = zeros(size(puncta_set,6),params.NUM_ROUNDS);
transcripts_calls = zeros(length(center_pixels)^3,params.NUM_ROUNDS,size(puncta_set,6));
for p_idx = 1:size(puncta_set,6)
    puncta = puncta_set(:,:,:,:,:,p_idx);
    puncta_calls = zeros(PSIZE,PSIZE,PSIZE,params.NUM_CHANNELS);
    for rnd_idx = 1:params.NUM_ROUNDS
        
        puncta_exp = squeeze(puncta(:,:,:,rnd_idx,:));
        [vmax,cmax] = max(puncta_exp,[],4);
        puncta_calls(:,:,:,rnd_idx) = cmax;
        
        %         for z_idx = 1:PSIZE
        %             output_img((rnd_idx-1)*PSIZE+1:rnd_idx*PSIZE,...
        %                 (z_idx-1)*PSIZE+1:z_idx*PSIZE) = squeeze(puncta_calls(:,:,z_idx,rnd_idx));
        %         end
    end
    
    calls = zeros(length(center_pixels)^3,params.NUM_ROUNDS);
    
    row_ctr = 1;
    %For each round, collapse (x,y,z,c) into (x,y,z) and pix val is the max
    %color channel
    for z_idx =center_pixels
        for y_idx =center_pixels
            for x_idx =center_pixels
                calls(row_ctr,:) = reshape(puncta_calls(x_idx,y_idx,z_idx,:),1,[]);
                row_ctr = row_ctr+1;
            end
        end
    end
    
    transcripts(p_idx,:) = mode(calls,1);
    transcripts_calls(:,:,p_idx)=calls;
    %For each base, what was the percentage of agreement
    transcripts_votes(p_idx,:) = sum(calls==transcripts(p_idx,:),1)/size(calls,1);
    
    if mod(p_idx,1000) ==0
        fprintf('%i/%i called\n',p_idx,size(puncta_set,6));
    end
    
    %     figure(1)
    %     makeImageOfPunctaROIs(puncta,transcripts(p_idx,:),params,1)
    %     figure;
    %     imagesc(calls)
    %     figure;
    %  output_img = zeros(PSIZE*params.NUM_ROUNDS,PSIZE*PSIZE);
    
    %     imagesc(output_img)
    
end


%% Randomize the pixels before visualizing
indices_to_view = randperm(size(transcripts,1),100);
%% Visualize a handful of results
if 0
    hasInitGif=0;
%     params.reportingDir = '/Users/Goody/Neuro/ExSeq/splintrseqroi1';
    giffilename= fullfile(params.reportingDir,sprintf('%s_gridplotdemo.gif',params.FILE_BASENAME));
    for x = indices_to_view
        puncta = puncta_set(:,:,:,:,:,x);
        makeImageOfPunctaROIs(puncta, transcripts(x,:),params,1);
        
        drawnow
        if hasInitGif==0
            pause
        end
        
        frame1 = getframe(1);
        im1 = frame2im(frame1);
        
        
        
        [imind1,cm1] = rgb2ind(im1,256);
        
        if hasInitGif==0
            imwrite(imind1,cm1,giffilename,'gif', 'Loopcount',inf);
            hasInitGif = 1;
        else
            imwrite(imind1,cm1,giffilename,'gif','WriteMode','append');
            
        end
        
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

%% Make sets of transcripts and create a new transcript object

%AS A TEMPORARY HACK, only process the uniques
%In the long run this will be helpful too when implemented fully
%[transcripts, ia, ic] = unique(transcripts,'rows');

%Size 17 because we score from round 4-20
round_mask = ones(1,17);
round_mask(11) = 0; %ignore round 14
round_mask(2) = 0; %ignore round 5
round_mask = logical(round_mask);

transcript_objects = cell(size(transcripts,1),1);
for p_idx = 1:size(transcripts,1)
    
    transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts(p_idx,4:end);
    %Sanity check: randomize the img_transcript

    %img_transcript = img_transcript(randperm(length(img_transcript)));

 
    %Search for a perfect match in the ground truth codes
%     hits = (groundtruth_codes==img_transcript);
    hits = (groundtruth_codes(:,round_mask)==img_transcript(round_mask));    

    %Calculate the hamming distance (now subtracting primer length)
    scores = length(img_transcript)- sum(hits,2) - sum(~round_mask);
    [values, indices] = sort(scores,'ascend');
    
    %Get the first index that is great than the best score
    %If this is 1, then there is a best fit
    idx_last_tie = find(values>values(1),1)-1;
    
    %Assuming the groundtruth options are de-duplicated
    %Is there a perfect match to the (unique ) best-fit
    if idx_last_tie==1
        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_votes=transcripts_votes(p_idx,:);
        transcript.img_transcript_calls=transcripts_calls(:,:,p_idx);
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
                sum( 1-transcripts_votes(...
                img_transcript== groundtruth_codes(indices(idx),:)...
                ,:));
        end
                
        [val_best_second_place,idx_best_second_place]=max(metrics_difference);

        transcript.img_transcript=transcripts(p_idx,:);
        transcript.img_transcript_votes=transcripts_votes(p_idx,:);
        transcript.img_transcript_calls=transcripts_calls(:,:,p_idx);
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


save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches.mat',params.FILE_BASENAME)),'transcript_objects','-v7.3');

%%

error_reads = zeros(17,1);
indices_for_specific_distance = []; ctr = 1;
for idx = 1:size(transcript_objects,1)
    if transcript_objects{idx}.distance_score-3==3
                indices_for_specific_distance(ctr) = idx;
                ctr = ctr +1;
    end
    addition = (transcript_objects{idx}.img_transcript(4:end)~=transcript_objects{idx}.known_sequence_matched);
    error_reads = error_reads + double(addition');
end

puncta_set_cropped =  puncta_set(:,:,:,:,:,indices_for_specific_distance);
save(fullfile(params.transcriptResultsDir,sprintf('%s_RANDpunctanormed_crop3.mat',params.FILE_BASENAME)),'puncta_set_cropped','-v7.3');
