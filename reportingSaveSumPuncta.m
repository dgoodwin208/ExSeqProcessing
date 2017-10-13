loadParameters;
chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

for round_num = 1:params.NUM_ROUNDS
    for chan_num = 1:params.NUM_CHANNELS
        chan_str = chan_strs{chan_num};
        filename_in = fullfile(params.punctaSubvolumeDir,sprintf('%s_round%.03i_%s_puncta.tif',params.FILE_BASENAME,round_num,chan_str));
        stack_in = load3DTif_uint16(filename_in);
        rounds(:,:,:,chan_num)=stack_in;
    end
    sum_rounds = sum(rounds,4);
    
    filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_round%.03i_summedpuncta.tif',params.FILE_BASENAME,round_num));
    save3DTif_uint16(sum_rounds,filename_out);
end

 %% Similarly, do the same thing but JUST of centroids
 imgSize = size(stack_in);
 sum_img = zeros(imgSize);
for round_num = 1:params.NUM_ROUNDS
    fprintf('Round %i\n',round_num);
    img = makeDebugImageOfPoints(puncta_centroids{round_num}(:,[2 1 3]),imgSize)/2;
    filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_round%.03i_centroids.tif',params.FILE_BASENAME,round_num));
    save3DTif_uint16(img,filename_out);
    sum_img = sum_img+img;
end

filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_ALLcentroidsSummed.tif',params.FILE_BASENAME));
save3DTif_uint16(sum_img,filename_out);

%% Create images of all the pixels from the punctafeinder per round

for round_num = 1:params.NUM_ROUNDS
    fprintf('Round %i\n',round_num);
    img = zeros(imgSize);
    
    for i= 1:length(puncta_voxels{round_num})
        indices_to_add = puncta_voxels{round_num}{i};
        img(indices_to_add)=10;
    end
     
    filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_round%.03i_punctapixels.tif',params.FILE_BASENAME,round_num));
    save3DTif_uint16(img,filename_out);
end

%% Create an image that is the sum of all puncta ROIs
 imgSize = size(stack_in);
 sum_img = zeros(imgSize);
for round_num = 3:params.NUM_ROUNDS
    fprintf('Round %i\n',round_num);

    
     for i= 1:length(puncta_voxels{round_num})
            %Set the pixel value to somethign non-zero
        indices_to_add = puncta_voxels{round_num}{i};
        sum_img(indices_to_add)=sum_img(indices_to_add)+1;
     end


end

filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_ALLPixelIdxListSummedNoPrimer.tif',params.FILE_BASENAME));
save3DTif_uint16(sum_img,filename_out);

%% Create an image that is the sum of all exclusive path ROIs

 imgSize = size(stack_in);
 sum_img = zeros(imgSize);
for round_num = 3:params.NUM_ROUNDS
    fprintf('Round %i\n',round_num);

    
     for i= 2:size(exclusive_paths,1)
            %Set the pixel value to somethign non-zero
        indices_to_add = puncta_voxels{round_num}{exclusive_paths(i,round_num)};
        sum_img(indices_to_add)=sum_img(indices_to_add)+1;
     end

end

filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_exclusivePathROIs.tif',params.FILE_BASENAME));
save3DTif_uint16(sum_img,filename_out);

%% For each individual path, generate an image sequentially

imgSize = size(stack_in);
sum_img = zeros(imgSize);
figure;
for i= size(exclusive_paths,1):-1:2
    for round_num = 3:params.NUM_ROUNDS
        indices_to_add = puncta_voxels{round_num}{exclusive_paths(i,round_num)};
        sum_img(indices_to_add)=sum_img(indices_to_add)+1;
    end
    
    imagesc(max(sum_img,[],3));
    pause
end

