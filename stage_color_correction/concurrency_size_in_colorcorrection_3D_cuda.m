function max_run_jobs = concurrency_size_in_colorcorrection_3D_cuda()

    loadParameters;

    if ~isfield(params,'COLOR_CORRECTION_MAX_RUN_JOBS')
        [imgsize_dbl,dim] = imagesize_in_double(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.tif',params.FILE_BASENAME,params.CHAN_STRS{1})));
        downsample_imgsize_dbl = imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
        bead_downsample_imgsize_dbl = downsample_imgsize_dbl * (dim(3) - params.BEAD_ZSTART)/dim(3);
        max_availablemem = availablememory();
        availablemem = max_availablemem * params.USABLE_MEM_RATE;
        expected_mem_usage = 10*downsample_imgsize_dbl + 3*bead_downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_run_jobs = min(params.NUM_ROUNDS,uint32(availablemem / expected_mem_usage));

        fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',max_availablemem,availablemem);
        fprintf('## image size (double) = %6.1f MiB, downsampling image size (double) = %6.1f MiB\n',imgsize_dbl,downsample_imgsize_dbl);
        fprintf('## expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
    else
        max_run_jobs = params.COLOR_CORRECTION_MAX_RUN_JOBS;
    end
    fprintf('## COLOR_CORRECTION_MAX_RUN_JOBS  = %d\n',max_run_jobs);
    if ~isfield(params,'COLOR_CORRECTION_MAX_THREADS')
        fprintf('## COLOR_CORRECTION_MAX_THREADS = automatic\n');
    else
        fprintf('## COLOR_CORRECTION_MAX_THREADS = %d\n',params.COLOR_CORRECTION_MAX_THREADS);
    end


end
