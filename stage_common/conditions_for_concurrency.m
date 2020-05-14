function cond = conditions_for_concurrency()

    loadParameters;

    cond = struct;

    cond.dim = image_dimensions(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.%s',params.FILE_BASENAME,params.CHAN_STRS{1},params.INPUT_IMAGE_EXT)));
    cond.imgsize_dbl            = prod(cond.dim)*8 / 1024 / 1024;
    cond.imgsize_uint16         = cond.imgsize_dbl / 4;
    cond.imgsize_int8           = cond.imgsize_dbl / 8;
    cond.downsample_imgsize_dbl = cond.imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
    cond.max_availablemem       = availablememory();
    cond.availablemem           = cond.max_availablemem * params.USABLE_MEM_RATE;

    fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',cond.max_availablemem,cond.availablemem);
    fprintf('## image size (double) = %6.1f MiB, downsampled image size (double) = %6.1f MiB\n',cond.imgsize_dbl,cond.downsample_imgsize_dbl);
    fprintf('## image dimensions = [%d, %d, %d]\n',cond.dim(1),cond.dim(2),cond.dim(3));

end
