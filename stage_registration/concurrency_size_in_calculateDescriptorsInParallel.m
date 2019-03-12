function max_run_jobs = concurrency_size_in_calculateDescriptorsInParallel()

    loadParameters;

    if ~isfield(params,'CALC_DESC_MAX_RUN_JOBS')
        [imgsize_dbl,dim] = imagesize_in_double(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.tif',params.FILE_BASENAME,params.CHAN_STRS{1})));
        downsample_imgsize_dbl = imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
        max_availablemem = availablememory();
        availablemem = max_availablemem * params.USABLE_MEM_RATE;
        if params.USE_GPU_CUDA
            expected_mem_usage = 10 * downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        else
            expected_mem_usage = 12 * downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        end
        max_run_jobs = min(params.NUM_ROUNDS,uint32(availablemem / expected_mem_usage));

        fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',max_availablemem,availablemem);
        fprintf('## image size (double) = %6.1f MiB, downsampling image size (double) = %6.1f MiB\n',imgsize_dbl,downsample_imgsize_dbl);
        fprintf('## expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
    else
        max_run_jobs = params.CALC_DESC_MAX_RUN_JOBS;
    end

    fprintf('## CALC_DESC_MAX_RUN_JOBS = %d\n',max_run_jobs);
end
