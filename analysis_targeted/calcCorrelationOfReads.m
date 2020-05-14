function [reads1_common,reads2_common,strings_common] = calcCorrelationOfReads(reads1,strings1,reads2,strings2)
%calcCorrelationOfReads Calculate the correlation between two vectors of
%genes 

%Get the list of all genes present
strings_common = union(strings1,strings2);

%Map reads1 into the common list of genes now
reads1_common = zeros(length(strings_common),1);
for r = 1:length(reads1)
    IndexC = strcmp(strings_common,strings1{r}); %returns a cell array of hits
    Index = find(IndexC);
    reads1_common(Index) = reads1(r);
end

%Map reads1 into the common list of genes now
reads2_common = zeros(length(strings_common),1);
for r = 1:length(reads2)
    IndexC = strcmp(strings_common,strings2{r}); %returns a cell array of hits
    Index = find(IndexC);
    reads2_common(Index) = reads2(r);
end


end

