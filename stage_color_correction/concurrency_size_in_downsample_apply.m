function max_pool_size = concurrency_size_in_downsample_apply()

    loadParameters;

    if ~isfield(params,'DOWN_SAMPLING_MAX_POOL_SIZE')
        [imgsize_dbl,dim] = imagesize_in_double(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.tif',params.FILE_BASENAME,params.CHAN_STRS{1})));
        downsample_imgsize_dbl = imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
        max_availablemem = availablememory();
        availablemem = max_availablemem * params.USABLE_MEM_RATE;
        expected_mem_usage = 2 * imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_pool_size = min(params.NUM_ROUNDS,uint32(availablemem / expected_mem_usage));

        fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',max_availablemem,availablemem);
        fprintf('## image size (double) = %6.1f MiB, downsampling image size (double) = %6.1f MiB\n',imgsize_dbl,downsample_imgsize_dbl);
        fprintf('## expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
    else
        max_pool_size = params.DOWN_SAMPLING_MAX_POOL_SIZE;
    end

    fprintf('## DOWN_SAMPLING_MAX_POOL_SIZE = %d\n',max_pool_size);
end
