loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    fprintf('[DONE]\n');
    return;
end

for rnd_indx = 1:params.NUM_ROUNDS
    precheck = true;
    for c = 1:params.NUM_CHANNELS

        filename_full = fullfile(params.deconvolutionImagesDir,...
            sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c}));

        if ~exist(filename_full,'file')
            precheck = false;
            fprintf('[ERROR] missing file: %s\n', filename_full);
        end
    end

    if ~precheck
        exit
    end
end


parpool(params.DOWN_SAMPLING_MAX_POOL_SIZE);

parfor rnd_indx = 1:params.NUM_ROUNDS
    for c = 1:params.NUM_CHANNELS

        filename_full = fullfile(params.deconvolutionImagesDir,...
            sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c}));
        filename_downsampled = fullfile(params.deconvolutionImagesDir,...
            sprintf('%s-downsample_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c},params.IMAGE_EXT));
        filename_full_hdf5 = replace(filename_full,'tif','h5');

        if exist(filename_downsampled,'file')
            if isequal(params.IMAGE_EXT,'tif')
                fprintf('Skipping file %s that already exists\n',filename_downsampled);
                continue;
            elseif isequal(params.IMAGE_EXT,'h5') && exist(filename_full_hdf5,'file')
                fprintf('Skipping files %s and %s that already exist\n',filename_downsampled,filename_full_hdf5);
                continue;
            end
        end

        img = load3DTif_uint16(filename_full);

        %Doing 'linear' downsample because the default of cubic was creating
        %values as low as -76
        img_downsample = imresize3(img,1/params.DOWNSAMPLE_RATE,'linear');

        if ~exist(filename_downsampled,'file')
            fprintf('Saving %s \n',filename_downsampled);
            save3DImage_uint16(img_downsample,filename_downsampled);
        end

        if isequal(params.IMAGE_EXT,'h5') && ~exist(filename_full_hdf5,'file')
            fprintf('Saving %s as hdf5\n',filename_full);
            save3DImage_uint16(img,filename_full_hdf5);
        end

    end
end

