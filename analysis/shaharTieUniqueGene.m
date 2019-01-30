function [dictmatchidx,minEditDist] = shaharTieUniqueGene(transcript,confidence,dictionary,labels,confThresh,numDrops,editScoreMax)

%Note there is a hardcoded delimiter as a temporary hack
DELIMITER = char(9);

%Pass in the cropped transcript and confidences, ie excluding primer rounds
L = length(transcript);

%First, find the best match
hits = (dictionary==transcript);
hammings= L-sum(hits,2);
minEditDist = min(hammings);

%If the minimum edit distance is greater than the allowable max, 
%throw it away
if minEditDist>editScoreMax
    dictmatchidx=-1;
    return
end

%Get the best score
%min_hamming_score_indices = find(hammings==minEditDist);

%If we find a unique perfect match, we're done!
if minEditDist==0
    %dictmatchidx = min_hamming_score_indices;
    dictmatchidx = find(hammings==0);
    return;    
end

min_hamming_score_indices = find(hammings<=editScoreMax);

mask = confidence>=confThresh;
%If there are not enough puncta of quality reads
if sum(mask)< (L-numDrops)
    dictmatchidx=-1;
    return
end

% We'll only allow misses where the confidence is low
%hits is the boolean matrix of agreements between gt and transcript
candidate_hits = hits(min_hamming_score_indices,:);

allowable_mistakes = ~mask';
%The candidate_hit is 0 if it's a mismatch, and ~mask is 1 for bases 
%that allowed to be mismatches
for i = 1:size(candidate_hits,1)
    count_agreement_hamming_errors_and_conf(i) = ...
        sum(~candidate_hits(i,:) & (allowable_mistakes));  
end
 
%To be a keepable option, each of the mismatches has to occur at an
%allowable base. It's a weird implementation above but we're summing up
%each mismatch that is also an allowable base, and confirm here that they
%must add up to the total hamming distance
keepable_indices = find(count_agreement_hamming_errors_and_conf == min(hammings(min_hamming_score_indices)));

%If we can't get an agreement between mismatches and allowable mismatch
%locations, return
if ~any(keepable_indices)
    dictmatchidx=-1;
    return
end

%Further refine the list to the minimum distance after applying the filter of allowable bases to edit
minEditDist = min(hammings(min_hamming_score_indices));
%keepable_indices = keepable_indices & (hammings(min_hamming_score_indices)==minFinalScore);

keepable_options = min_hamming_score_indices(keepable_indices);
candidate_transcript_hitlabels = labels(keepable_options);

%If there is just one remaining option, use that!
if length(keepable_options)==1
    dictmatchidx = keepable_options(1);
    return;
end

%And if there are multiple possibilities, explore the similarity of
%alignment information

gene_symbols = cell(length(keepable_options),1);
refseq_codes = cell(length(keepable_options),1);


for oidx = 1:length(keepable_options)
   parts = split(candidate_transcript_hitlabels{oidx},DELIMITER);
   if length(parts)>=3 %then it's a refseq hit
       gene_symbols{oidx} = parts{2};
   end
   
   %If it's not a refseq hit, then let's see if it's a refseq code or
   %genomic hit
   match_id_parts = split(parts{1},'_');
   if length(match_id_parts)==2
      %this is a NoAlignment hit, sp nothign to note
   elseif match_id_parts{2}(1)=='N'
       refseq_prefix = match_id_parts{2};
       refseq_integer = floor(str2double(match_id_parts{3}));
       refseq_codes{oidx} = sprintf('%s_%i',refseq_prefix,refseq_integer);
   end
end

%Now use gene_symbols and refseq_codes to see if we have an agreement
gene_symbols_nonempty = gene_symbols(~cellfun(@isempty,gene_symbols));
gene_symbols_unique =unique(gene_symbols_nonempty,'stable');

%If we have one unique gene symbol, we use that
if length(gene_symbols_unique)==1
    idx_gene_symbol = find(~cellfun(@isempty,gene_symbols));
    dictmatchidx = keepable_options(idx_gene_symbol(1));
    %fprintf('Confirming %s = %s\n', labels{dictmatchidx},candidate_transcript_hitlabels{idx_gene_symbol(1)});
    return;
elseif length(gene_symbols_unique)>1%If we have multiple different gene symbols, we're in trouble
    dictmatchidx=-1;
    return;
end

%Otherwise, compare the similarity of the refseq numbers
refseq_nonempty = refseq_codes(~cellfun(@isempty,refseq_codes));
refseq_unique =unique(refseq_nonempty,'stable');

%If we have one unique refseq code, we use that
if length(refseq_unique)==1
    idx_refseq_code = find(~cellfun(@isempty,refseq_codes));
    dictmatchidx = keepable_options(idx_refseq_code(1));
    %fprintf('Confirming %s = %s\n', labels{dictmatchidx},candidate_transcript_hitlabels{idx_refseq_code(1)});
    return;
elseif length(refseq_unique)>1
    dictmatchidx=-1;
    return;
end

% Now we check the genomic hits. We remove any candidates that have NoAlignment in their label. If there's a unique element left, that's our choice.
IndexC = strfind(candidate_transcript_hitlabels,'NoAlignment');
Index = find(cellfun('isempty', IndexC));
if length(Index)==1
    %fprintf('Ignoring NoAlignments to get hit: %s\n',strjoin(candidate_transcript_hitlabels));
    dictmatchidx = keepable_options(Index);
    return;
else 
     %fprintf('Discaring because conflicting genomic hits: %s\n',strjoin(candidate_transcript_hitlabels));
    dictmatchidx = -1;
    return;
end


return



