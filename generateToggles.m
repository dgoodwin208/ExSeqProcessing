function [ vector_variants ] = generateToggles(vector, indices_toggle, options_toggle)
%GENERATETOGGLES Make lots of variants of a vector
%   Indices_toggle is a vector of all positions to toggle
%   options_toggle is a vector of all options per position of
%   indices_toggle


%Call this function recursively

%If there is only one index to toggle, create the two options and return
if length(indices_toggle)==1
    
    %    return the two options
    option1 = vector; option2 = vector;
    option1(indices_toggle(1)) = options_toggle{1}(1);
    option2(indices_toggle(1)) = options_toggle{1}(2);
    vector_variants = cell(2);
    vector_variants{1} = option1; vector_variants{2} = option2;
    
else
    
    all_options = generateToggles(vector, indices_toggle(2:end),options_toggle(2:end));
    
    %Now make the toggles to the first index
    %Duplicate the output and just change the first index
    other_toggle = all_options;
    for idx = 1:length(other_toggle)
        candidate_transcript = other_toggle{idx};
        candidate_transcript(indices_toggle(1)) = options_toggle{1}(2);
        other_toggle{idx} = candidate_transcript;
    end
    
    
    vector_variants = cell(1,length(all_options)*2);
    for idx = 1:length(other_toggle)
        vector_variants(2*(idx-1)+1) = all_options(idx);
        vector_variants(2*idx) = other_toggle(idx);
    end
end

end





