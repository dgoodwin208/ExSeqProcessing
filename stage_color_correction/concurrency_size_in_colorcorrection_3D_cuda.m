function max_run_jobs = concurrency_size_in_colorcorrection_3D_cuda(cond)

    loadParameters;

    if ~isfield(params,'COLOR_CORRECTION_MAX_RUN_JOBS')
        bead_downsample_imgsize_dbl = cond.downsample_imgsize_dbl * (cond.dim(3) - params.BEAD_ZSTART)/cond.dim(3);

        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        job_worker_mem_usage = 10*cond.downsample_imgsize_dbl + 3*bead_downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_run_jobs = min(params.NUM_ROUNDS,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        total_mem_usage = main_mem_usage + job_worker_mem_usage*max_run_jobs;

        fprintf('## total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
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
