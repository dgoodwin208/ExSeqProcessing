
loadParameters;
params.basecallingResultsDir = '/Users/Goody/Neuro/ExSeq/exseq20170524/6_base-calling';
%% Load Ground truth information
%load(fullfile(params.basecallingResultsDir,'groundtruth_codes.mat'));
%Turn the Nx3 array into a set of strings
%gtlabels = {};
%for i = 1:size(groundtruth_codes,1)
%    transcript_string = '';
%    for c = 1:size(groundtruth_codes,2)
%        transcript_string(c) = num2str(groundtruth_codes(i,c));
%    end
%    gtlabels{i}=transcript_string;
%end
t = fastaread(fullfile(params.basecallingResultsDir, 'SOLiD_like_reads_in_Illumina_unique_17.fa'));

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

%% Convert from string to base

transcripts = zeros(size(basecallclosest,2),params.NUM_ROUNDS);

for p_idx = 1:size(transcripts,1)
    for r_idx = 4:params.NUM_ROUNDS
        transcripts(p_idx,r_idx) = str2double(basecallclosest(r_idx,p_idx));
    end
end

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


%% Quick and dirty scoring of transcripts 

hamming_scores = zeros(size(transcripts,1),1);
% par_factor = 5;

%Size 17 because we score from round 4-20
round_mask = ones(1,17);
round_mask(11) = 0; %ignore round 14
round_mask(2) = 0; %ignore round 5
round_mask = logical(round_mask);


for p_idx = 1:size(transcripts,1)
    
    
%     transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts(p_idx,4:end);
%     img_transcript = [transcripts(p_idx,:) 0];
    %Sanity check: randomize the img_transcript
 
    %NEW: updating the round 
    round_mask = (img_transcript ~= 0);
    
    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes(:,round_mask)==img_transcript(round_mask));
    

    %Calculate the hamming distance 
    scores = length(img_transcript)- sum(hits,2) - sum(~round_mask);
%     [values, indices] = sort(scores,'ascend');    
%     hamming_scores(p_idx) = values(1);
    hamming_scores(p_idx) = min(scores);
    
    if mod(p_idx,500) ==0
        fprintf('%i/%i matched\n',p_idx,size(transcripts,1));
    end
end

hamming_clipped = hamming_scores(1:p_idx-1);
figure;
histogram(hamming_clipped,length(unique(hamming_clipped)))
