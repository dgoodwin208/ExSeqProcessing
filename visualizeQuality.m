%  This script visualized the quality of the transcripts and creates a mask
%  that can be applied to puncta to remove registration+experimentnal
%  errors in the base calling

%% Load all necessary data

%load the flags_vector mat file
load('slice.mat'); 
%Load transcript files that contain pos, transcripts,
%transcripts_confidence
load('transcripts_center-only.mat');

IMG_SIZE = [2048,2048];

%%
colors = {'r.','g.','b.','y.','c*'};
figure;
hold on;

for class_type=1:5
    I = find(flags_vector==class_type);
    plot(pos(I,2),pos(I,1),colors{class_type},'MarkerSize',10);
end
legend('low confidence','low expression', ...
    'low confidence+expression', 'No known barcode', 'Good match');
hold off;

%% Plot all in 3D

colors = {'r.','r.','r.','g.','b*'};

figure;
hold on;
for class_type=1:5
    I = find(flags_vector==class_type);
    plot3(pos(I,2),pos(I,1),pos(I,3),colors{class_type},'MarkerSize',10);
end
legend('low confidence','low expression', ...
    'low confidence+expression', 'No known barcode', 'Good match');
hold off;

%% Plot each class in a separate 3D plot

colors = {'r.','r.','r.','g.','b*'};
figure;
hold on;
for class_type=1:3
    I = find(flags_vector==class_type);
    plot3(pos(I,2),pos(I,1),pos(I,3),colors{class_type},'MarkerSize',10);
end
hold off;

for class_type=4:5
    figure;
    I = find(flags_vector==class_type);
    
    
    plot3(pos(I,2),pos(I,1),pos(I,3),colors{class_type},'MarkerSize',10);
    
end

%% Make a heatmap

%First make a 3D matrix to contain all the class information: (X,Y,class)
classes = zeros(IMG_SIZE(1),IMG_SIZE(2),3);

%Index 1 will be the three "bad puncta" types:
%'low confidence','low expression', 'low confidence+expression'
for class_type=1:3
    I = find(flags_vector==class_type);
    for x=1:length(I)
        classes(pos(I(x),1),pos(I(x),2),1) = 1;
    end
end

%Index 2 will be the "bad read" type: 'No known barcode'
I = find(flags_vector==4);
for x=1:length(I)
    classes(pos(I(x),1),pos(I(x),2),2) = 1;
end

%Index 3 is the good transcript type
I = find(flags_vector==5);
for x=1:length(I)
    classes(pos(I(x),1),pos(I(x),2),3) = 1;
end

%for every x,y coordinate, get a basic statistic on the
%good reads/bad reads ratio
%define a neighborhood from which we draw a square of size 2n x 2n
n = 50;

%Use convolution to create heatmaps of the three class types
summing_filter = ones(2*n,2*n);
reads_bad = conv2(classes(:,:,1),summing_filter,'same') + conv2(classes(:,:,2),summing_filter,'same');
reads_mis = conv2(classes(:,:,2),summing_filter,'same');
reads_good =  conv2(classes(:,:,3),summing_filter,'same');

%Create a 3D matrix as if it was an RGB image for visualization convenience
quality_map = zeros(IMG_SIZE(1),IMG_SIZE(2),3);
quality_map(:,:,1) = reads_good./max(reads_bad+reads_mis,1);
quality_map(:,:,2) = reads_good + reads_bad+reads_mis;


%rescale the two color dimensions
figure;
subplot(1,3,1);
rgb_img = zeros(size(quality_map));
rgb_img(:,:,1) = quality_map(:,:,1)/max(max(quality_map(:,:,1)));
imshow(rgb_img);
title('Good reads / (bad reads + mis-reads)');

subplot(1,3,2);
rgb_img = zeros(size(quality_map));
rgb_img(:,:,2) = quality_map(:,:,2)/max(max(quality_map(:,:,2)));
imshow(rgb_img);
title(sprintf('Total puncta in a %i neighborhood',n));

subplot(1,3,3);
rgb_img(:,:,1) = quality_map(:,:,1)/max(max(quality_map(:,:,1)));
imshow(rgb_img);
title('Combined into an RGB image');

%% Make a mask of read quality
%A value of 1 in qualitymap(:,:,1) means there is only one good puncta in the
%neighborhood or there are more bad puncta/reads than good 

THRESHOLD = 1;
read_quality_mask = quality_map(:,:,1)>1;
figure; 
imagesc(read_quality_mask);
title('Read quality mask');
save('read_quality_mask.mat','read_quality_mask');