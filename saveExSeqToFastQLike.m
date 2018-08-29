function [Fastqstructs] = saveExSeqToFastQLike(transcripts,confidences,headers)
%SAVEEXSEQTOFASTQLIKE 
% Write to standard sequencing format for analysis in FASTQC etc.

%initialize the fastq structs
Fastqstructs(size(transcripts,1)) = struct;
mapping_base = 'ACGT';
mapping_quality = '!"#$%&''()*+,-./0123456789:;<=>?@ABCDEFGHI';

readlength = size(transcripts,2);

for idx = 1:size(transcripts,1)
    seq = blanks(readlength);
    quality = blanks(readlength);
    quality_number = round(confidences(idx,:));
    quality_number(quality_number>40) = 40; %cap the score at 40;
    for c = 1:length(seq)
        seq(c) = mapping_base(transcripts(idx,c)); 
        quality(c) = mapping_quality(quality_number(c));
    end
    
    Fastqstructs(idx).Header = headers{idx};
    Fastqstructs(idx).Sequence = seq;
    Fastqstructs(idx).Quality = quality;
end

end


