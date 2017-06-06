load(fullfile(params.transcriptResultsDir,'transcriptsv9_punctameannormed.mat'));
load(fullfile(params.transcriptResultsDir,'groundtruth_codes.mat'));

%% Load ground truth codes

%Turn the Nx3 array into a set of strings
gtlabels = {};
for i = 1:size(groundtruth_codes,1)
    transcript_string = '';
    for c = 1:size(groundtruth_codes,2)
        transcript_string(c) = num2str(groundtruth_codes(i,c));
    end
    gtlabels{i}=transcript_string;
end

%% Get the number of unique codes present

[unique_codes, ia, ic] = unique(transcripts,'rows');

barcodes_count = size(unique_codes,1);

hist_counts = zeros(1,barcodes_count);

for i = 1:length(ic)
    hist_counts(ic(i)) = hist_counts(ic(i))+1;
end

[values, indices] = sort(hist_counts,'descend');

figure;

bar(values);

labels = {};
for i = 1:barcodes_count
    transcript_string = '';
    for c = 1:size(unique_codes,2)
        transcript_string(c) = num2str(unique_codes(i,c));
    end
    labels{i}=transcript_string;
end

xlabel('Unique barcode index')
%plot it
bar(values);
% Place the text labels
labels_resort = {};
for i = 1:barcodes_count
    labels_resort{i} = labels{indices(i)};
end

xticklabel_rotate(1:barcodes_count,45,labels_resort,'interpreter','none')
title(sprintf('Histogram of barcode distribution (unfiltered N=%i)',sum(hist_counts)));

%Now score this in terms of the alignment percentage to known barcodes

barcodes_count_correct = 0;
barcodes_count_incorrect = 0;
for i = 1:barcodes_count
    IndexC = strfind(gtlabels,labels{i});
    Index = find(not(cellfun('isempty', IndexC)));
    
    if ~isempty(Index)
       barcodes_count_correct = barcodes_count_correct + hist_counts(i); 
    else
        barcodes_count_incorrect = barcodes_count_incorrect + hist_counts(i);
    end
end

fprintf('not-filtered call: Barcodes valid: %i, Barcodes invalid: %i\n',...
    barcodes_count_correct,...
    barcodes_count_incorrect);


%% Using Shahar's 

[unique_transcripts, ia, ic] = unique(transcripts_filtered,'rows');



barcodes_count = size(unique_transcripts,1);

labels = {};
for i = 1:barcodes_count
    transcript_string = '';
    for c = 1:size(unique_transcripts,2)
        transcript_string(c) = num2str(unique_transcripts(i,c));
    end
    labels{i}=transcript_string;
end
    
hist_counts = zeros(1,barcodes_count);

for i = 1:length(ic)
    hist_counts(ic(i)) = hist_counts(ic(i))+1;
end

[values, indices] = sort(hist_counts,'descend');

figure;

%plot it
bar(values);
% Place the text labels
labels_resort = {};
for i = 1:barcodes_count
    labels_resort{i} = labels{indices(i)};
end

xticklabel_rotate(1:barcodes_count,45,labels_resort,'interpreter','none')
title(sprintf('Histogram of barcode distribution after confidence filtering, N=%i',sum(hist_counts)));

barcodes_count_correct = 0;
barcodes_count_incorrect = 0;
for i = 1:barcodes_count
    IndexC = strfind(gtlabels,labels{i});
    Index = find(not(cellfun('isempty', IndexC)));
    
    if ~isempty(Index)
       barcodes_count_correct = barcodes_count_correct + hist_counts(i); 
    else
        barcodes_count_incorrect = barcodes_count_incorrect + hist_counts(i);
    end
end

fprintf('filtered call: Barcodes valid: %i, Barcodes invalid: %i, Accuracy %.02f\n',...
    barcodes_count_correct,...
    barcodes_count_incorrect,...
    barcodes_count_correct/(barcodes_count_correct+barcodes_count_incorrect))

%% Using a combination of intra- and inter-color comparisons

[unique_transcripts, ia, ic] = unique(transcripts_probfiltered,'rows');



barcodes_count = size(unique_transcripts,1);

labels = {};
for i = 1:barcodes_count
    transcript_string = '';
    for c = 1:size(unique_transcripts,2)
        transcript_string(c) = num2str(unique_transcripts(i,c));
    end
    labels{i}=transcript_string;
end
    
hist_counts = zeros(1,barcodes_count);

for i = 1:length(ic)
    hist_counts(ic(i)) = hist_counts(ic(i))+1;
end

[values, indices] = sort(hist_counts,'descend');

figure;

%plot it
bar(values);
% Place the text labels
labels_resort = {};
for i = 1:barcodes_count
    labels_resort{i} = labels{indices(i)};
end

xticklabel_rotate(1:barcodes_count,45,labels_resort,'interpreter','none')
title(sprintf('Histogram of barcode distribution after probalistic filtering, N=%i',sum(hist_counts)));

barcodes_count_correct = 0;
barcodes_count_incorrect = 0;
for i = 1:barcodes_count
    IndexC = strfind(gtlabels,labels{i});
    Index = find(not(cellfun('isempty', IndexC)));
    
    if ~isempty(Index)
       barcodes_count_correct = barcodes_count_correct + hist_counts(i); 
    else
        barcodes_count_incorrect = barcodes_count_incorrect + hist_counts(i);
    end
end

fprintf('filtered call: Barcodes valid: %i, Barcodes invalid: %i. Accuracy: %.02f\n',...
    barcodes_count_correct,...
    barcodes_count_incorrect,...
    barcodes_count_correct/(barcodes_count_correct+barcodes_count_incorrect));


%% Using a combination of intra- and inter-color comparisons
% USED FOR EXSEQ 
BARCODE_SIZE = 5;
%There is a mapping of experimental colors to bowtie
CellArray={'0' '2' '1' '3'};

[unique_transcripts, ia, ic] = unique(transcripts_probfiltered(:,1:BARCODE_SIZE),'rows');



barcodes_count = size(unique_transcripts,1);

labels = {};
for i = 1:barcodes_count
    transcript_string = '';
    for c = 1:BARCODE_SIZE
        experimental_called_base = unique_transcripts(i,c);
        bowtie_called_base = CellArray{experimental_called_base};
        transcript_string(c) = num2str(bowtie_called_base);
    end
    
    labels{i}=transcript_string;
end
    
hist_counts = zeros(1,barcodes_count);
for i = 1:length(ic)
    hist_counts(ic(i)) = hist_counts(ic(i))+1;
end
[values, indices] = sort(hist_counts,'descend');

figure;
bar(values); %plot it
labels_resort = {}; % Place the text labels
for i = 1:barcodes_count
    labels_resort{i} = labels{indices(i)};
end
xticklabel_rotate(1:barcodes_count,45,labels_resort,'interpreter','none')
title(sprintf('Histogram of barcode distribution after probalistic filtering, N=%i',sum(hist_counts)));

barcodes_count_correct = 0;
barcodes_count_incorrect = 0;
for i = 1:barcodes_count
    barcode_candidate = labels{i};
    IndexC = strfind(gtlabels,barcode_candidate);
    Index = find(not(cellfun('isempty', IndexC)));
    
    if ~isempty(Index)
       barcodes_count_correct = barcodes_count_correct + hist_counts(i); 
    else
        barcodes_count_incorrect = barcodes_count_incorrect + hist_counts(i);
    end
end

fprintf('filtered call: Barcodes valid: %i, Barcodes invalid: %i. Accuracy: %.02f\n',...
    barcodes_count_correct,...
    barcodes_count_incorrect,...
    barcodes_count_correct/(barcodes_count_correct+barcodes_count_incorrect));

