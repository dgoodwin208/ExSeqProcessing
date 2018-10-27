%% Define the rounds
rounds_zoomed = [4,20,25,30];
rounds_norm = 1:30; rounds_norm(rounds_zoomed) = [];
DIRECTORY = '/Users/Goody/Neuro/ExSeq/exseq-pub/input/';
files = dir(fullfile(DIRECTORY,'*.tif'));
ZOOM_FACTOR = 1.5;
FIXED_STRING = 'fixed';

%% Load the data that we know represents two examples
zoomed_img = fullfile(DIRECTORY,'exseqauto_round4_summedNorm.tif');
imgz = load3DTif(zoomed_img);
maxz = max(imgz,[],3);

norm_img = fullfile(DIRECTORY,'exseqauto_round5_summedNorm.tif');
imgn = load3DTif(norm_img);
maxn = max(imgn,[],3);



crop_size = [round(size(maxz,1)/ZOOM_FACTOR), round(size(maxz,2)/ZOOM_FACTOR)];

top_left_x = round((size(maxz,1) - crop_size(1))/2) +1;
top_left_y = round((size(maxz,2) - crop_size(2))/2) +1;

imcrop_sample = imcrop(maxn,[top_left_x,top_left_y,crop_size(1)-1,crop_size(2)-1]);

imresize_sample = imresize(maxz,crop_size(1)/size(maxz,1));

figure;
subplot(1,2,1)
imagesc(imcrop_sample)
title('cropped normal image');

subplot(1,2,2)
imagesc(imresize_sample)
title('resized zoom');

disp('And what are the xy sizes of the output? (should be the same)')
size(imcrop_sample)
size(imresize_sample)
%% output the images
% 
% imgn_crop = zeros(size(imcrop_sample,1),size(imcrop_sample,2),size(imgn,3));
% for z = 1:size(imgn,3)
%     imgn_crop(:,:,z) = imcrop(squeeze(imgn(:,:,z)),[top_left_x,top_left_y,crop_size(1)-1,crop_size(2)-1]);
% end
% % save3DTif(imgn_crop, '/Users/Goody/Neuro/ExSeq/exseq-pub/input/exseqauto_round5_summedNormCrop.tif');
% 
% imgz_resize = zeros(size(imresize_sample,1),size(imresize_sample,2),size(imgz,3));
% for z = 1:size(imgz,3)
%     imgz_resize(:,:,z) = imresize(imgz(:,:,z),...
%         crop_size(1)/size(maxz,1));
% end
% save3DTif(imgz_resize, '/Users/Goody/Neuro/ExSeq/exseq-pub/input/exseqauto_round4_summedNormCropResize.tif');

%% Run for all rounds

for file_indx = 1:length(files)
    %quickly check that we're not processing an already-fixed image
    if ~isempty(strfind(files(file_indx).name,FIXED_STRING))
        fprintf('Skipping %s\n',files(file_indx).name);
        continue;
    end
    
    parts = split(files(file_indx).name,'_');
    string_roundnumber= parts{2};
    roundnumber_parts = split(string_roundnumber,'round');
    roundnumber = str2num(roundnumber_parts{2});
    
    %output_string: get the part of the filename before the .tif
    parts_output = split(files(file_indx).name,'.');
    
    outputfilename = strcat(fullfile(DIRECTORY,parts_output{1}),...
        'fixed','.tif');
    img = load3DTif(fullfile(DIRECTORY,files(file_indx).name));
    
    %the size of the imcrop and imresize should be the same
    img_out = zeros(size(imcrop_sample,1),size(imcrop_sample,2),size(img,3));
    
    if ismember(roundnumber,rounds_zoomed)
        %If it's a non-cropped data, then downsample the data 
        fprintf('Resizing %s by %f\n',files(file_indx).name,...
            crop_size(1)/size(maxz,1));
        
        for z = 1:size(img,3)
            img_out(:,:,z) = imresize(img(:,:,z),...
                crop_size(1)/size(maxz,1));
        end
        
        
    elseif ismember(roundnumber,rounds_norm)
        fprintf('Cropping %s to %i by %i\n',files(file_indx).name,...
            crop_size(1),crop_size(2));
        
        for z = 1:size(img,3)
            img_out(:,:,z) = imcrop(squeeze(img(:,:,z)),...
                [top_left_x,top_left_y,crop_size(1)-1,crop_size(2)-1]);
        end
    else
        disp('There is no reason we should have got here')
        barf()
    end
    
    fprintf('Now saving %s\n',outputfilename);
    
    save3DTif(img_out, outputfilename);
end