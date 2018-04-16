loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    return;
end

for rnd_indx = 1:params.NUM_ROUNDS
    for c = 1:params.NUM_CHANNELS
    
    filename_full = fullfile(params.deconvolutionImagesDir,...
        sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c}));
    filename_downsampled = fullfile(params.deconvolutionImagesDir,...
        sprintf('%s-downsample_round%.03i_%s.tif',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c}));
    
    if ~exist(filename_full,'file')
        fprintf('Skipping missing file %s \n',filename_full);
        continue;
    end
    
    if exist(filename_downsampled,'file')
        fprintf('Skipping file %s that already exists\n',filename_downsampled);
        continue;
    end
    
    img = load3DTif_uint16(filename_full);
    
    %Doing 'linear' downsample because the default of cubic was creating
    %values as low as -76
    img_downsample = imresize3(img,1/params.DOWNSAMPLE_RATE,'linear');
    
    fprintf('Saving %s \n',filename_downsampled);
    save3DTif_uint16(img_downsample,filename_downsampled);
    
    end
end
