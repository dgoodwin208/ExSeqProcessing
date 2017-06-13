loadParameters;

load(fullfile(params.punctaSubvolumeDir,'transcriptsv9_punctameannormed.mat'));
load(fullfile(params.rajlabDirectory,'transcripts_probfiltered.mat'));
% load(fullfile(params.punctaSubvolumeDir,'puncta_rois.mat'));

channel_mapping = [0,1,3,2];

%What round of sequencing is N-1P?
ALIGNMENT_START_IDX = 4;

transcripts_out = transcripts_probfiltered;
confidence_out = transcripts_probfiltered_probconfidence;

X_out = X(puncta_indices_probfiltered);
Y_out = Y(puncta_indices_probfiltered);
Z_out = Z(puncta_indices_probfiltered);

QUALITY_STRING = '!"#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJ';

max_score = length(QUALITY_STRING);

percentiles = prctile(-10*log10(confidence_out(:)),1:100);

min_conf = vals(end);
max_conf = vals(1)-min_conf;

for t_idx = 1:size(transcripts_out,1)
    %Get the transcript as an array of numbers
    t_out_num = squeeze(transcripts_out(t_idx,:));
    
    % Map the colors of the experiment to what bowtie expects
    %  blue=0, green=1, yellow=2 and red=3
    string_ctr = 1;
    quality_string = '';
    for base_idx = ALIGNMENT_START_IDX:length(t_out_num)
%         t_out_num(base_idx) = channel_mapping(t_out_num(base_idx));
        t_out_string(string_ctr) = num2str(channel_mapping(t_out_num(base_idx)));
        
        %Get the Phred score: -10log10(confidence)
        quality_score_num = -10*log10(confidence_out(t_idx,base_idx));
        %Find the nearest index in the percentiles vector, maxing at the 
        %length of the quality string
        if isinf(quality_score_num)
            quality_score_idx = length(QUALITY_STRING);
        else
            quality_score_idx = min(find(quality_score_num-percentiles<0,1),length(QUALITY_STRING));
        end
        quality_string(string_ctr) = QUALITY_STRING(quality_score_idx);
        
        string_ctr = string_ctr+1;
    end
    
    data(t_idx).Sequence = t_out_string;
    data(t_idx).Header = sprintf('Y=%i,X=%i,Z=%i',Y_out(t_idx),...
                                    X_out(t_idx),Z_out(t_idx));
    
    
    data(t_idx).Quality = quality_string;
end

fastafile_output = fullfile(params.punctaSubvolumeDir,'transcripts.csfasta');
if exist(fastafile_output,'file')
    delete(fastafile_output);
end
fastawrite(fastafile_output,data)

fastqfile_output = fullfile(params.punctaSubvolumeDir,'transcripts.csfastq');
if exist(fastqfile_output,'file')
    delete(fastqfile_output);
end
fastqwrite(fastqfile_output,data)
