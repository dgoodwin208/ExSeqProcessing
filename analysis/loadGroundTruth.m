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
