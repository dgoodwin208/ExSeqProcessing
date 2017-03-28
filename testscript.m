

figure;
subplot(3,1,1);
img = load3DTif(fullfile(params.rajlabDirectory ,'alexa001.tiff')); 
vec = img(:);
percentiles  = prctile(vec,[0,99.5]);
outlierIndex_bg = vec > percentiles(2);
vec(outlierIndex_bg) = [];
[values,binedges] = histcounts(vec,params.NUM_BUCKETS);
plot(binedges(1:params.NUM_BUCKETS),values);
title(sprintf('Experiment 1 for %s',params.rajlabDirectory))

subplot(3,1,2);
img = load3DTif(fullfile(params.rajlabDirectory ,'alexa002.tiff')); vec = img(:);
vec = img(:);
percentiles  = prctile(vec,[0,99.5]);
outlierIndex_bg = vec > percentiles(2);
vec(outlierIndex_bg) = [];
[values,binedges] = histcounts(vec,params.NUM_BUCKETS);
plot(binedges(1:params.NUM_BUCKETS),values);
title(sprintf('Experiment 2 for %s',params.rajlabDirectory))

subplot(3,1,3);
img = load3DTif(fullfile(params.rajlabDirectory ,'alexa003.tiff')); vec = img(:);
vec = img(:);
percentiles  = prctile(vec,[0,99.5]);
outlierIndex_bg = vec > percentiles(2);
vec(outlierIndex_bg) = [];
[values,binedges] = histcounts(vec,params.NUM_BUCKETS);
plot(binedges(1:params.NUM_BUCKETS),values);
title(sprintf('Experiment 3 for %s',params.rajlabDirectory))

%% Making ground truth from a cell array of strings
BARCODE_LENGTH = 5;
groundtruth_codes = zeros(length(All_Barcodes_colors),BARCODE_LENGTH);

for i = 1:size(groundtruth_codes,1)
    str = All_Barcodes_colors{i};
    for j = 1:BARCODE_LENGTH
        groundtruth_codes(i,j) = str2num(str(j));
    end

end

