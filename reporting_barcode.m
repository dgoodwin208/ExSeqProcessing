loadParameters;
load(fullfile(params.rajlabDirectory,'transcriptsv9_punctameannormed.mat'));
load(fullfile(params.rajlabDirectory,'groundtruth_codes.mat'));
load(fullfile(params.rajlabDirectory,'transcripts_probfiltered.mat'));

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

fprintf('Result: Barcodes valid: %i, Barcodes invalid: %i. Accuracy: %.02f\n',...
    barcodes_count_correct,...
    barcodes_count_incorrect,...
    barcodes_count_correct/(barcodes_count_correct+barcodes_count_incorrect));


labelstring_per_puncta = cell(length(transcripts_probfiltered),1);
for i = 1:length(labelstring_per_puncta)
    labelstring_per_puncta{i} = labels{ic(i)};
end

transcripts_string = labelstring_per_puncta;
transcripts_numerical = transcripts_probfiltered;
transcripts_position = pos(puncta_indices_probfiltered,:);
save(fullfile(params.rajlabDirectory,'barcodes_output.mat'),'transcripts_string','transcripts_numerical','transcripts_position');


