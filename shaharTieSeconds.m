function [dictmatchidx,minEditDist] = shaharTieSeconds(transcript,transcript2,confidence,dictionary,confThreshChange, confThreshFixed,editScoreMax)
%Pass in the cropped transcript and confidences, ie excluding primer rounds
L = length(transcript);

%First, find the best match
hits = (dictionary==transcript);
hammings= L-sum(hits,2);
minEditDist = min(hammings);

%Get the best score
min_hamming_score_indices = find(hammings==minEditDist);

%If we find a unique perfect match, we're done!
if minEditDist==0
    dictmatchidx = min_hamming_score_indices;
    minEditDist = 0;
    return
end


changeablebases = find(confidence<=confThreshChange);
unchangeablebases = find(confidence>confThreshFixed);

% We'll only allow misses where the confidence is low
candidate_transcript_hits = dictionary(min_hamming_score_indices,:);

%The candidate_hit is 0 if it's a mismatch, and changeablemask is 1 for
%bases that are under a confidene threshold that lets us change it to the
%second best fit

error_correctables_2ndmatch = zeros(length(min_hamming_score_indices),1);
for i = 1:size(candidate_transcript_hits,1)
    
    baseErrorsUnallowable = find(candidate_transcript_hits(i,unchangeablebases) ~= transcript(unchangeablebases));
    %We can't allow any errors for bases above a certain confidence
    %threshold
    if length(baseErrorsUnallowable)>0
        error_correctables_2ndmatch(i) = -1; %Signal that this match won't work
        continue;
    end
    
    baseErrors = find(candidate_transcript_hits(i,changeablebases) ~= transcript(changeablebases));
    
    %count all the bases in the ground truth that match the second
    %brightest channel from the in situ data
    for b = baseErrors
       if candidate_transcript_hits(i,changeablebases(b)) == transcript2(changeablebases(b))
            error_correctables_2ndmatch(i) = error_correctables_2ndmatch(i)+1;
        end
    end
end

%Which option has the best recovery using the second brightest base?
[num_correction, best_idx] = max(error_correctables_2ndmatch);

% If our best option has a change at an unallowable base, then we check if we can keep it
if max(num_correction)==-1
    %If there is a unique option for an allowable minimim edit distance, keep it!
    if length(min_hamming_score_indices)==1 && minEditDist <=editScoreMax
        dictmatchidx = min_hamming_score_indices;
        return;
    %But if we have two options that we can't tell apart, then we can't keep it
    else
         dictmatchidx=-1;
         return;
     end
end

%If we have multiple options for the min edit distance, we have to make sure
%there is a clear winner, otherwise we have to discard
if sum(error_correctables_2ndmatch==num_correction)>1
    dictmatchidx=-1;
    return;
end

% fprintf('Min error was %i, w correction: %i\n',minEditDist,minEditDist - num_correction);
minEditDist = minEditDist - num_correction;

if minEditDist<=editScoreMax
    dictmatchidx = min_hamming_score_indices(best_idx);
else
    dictmatchidx=-1;
end

return



