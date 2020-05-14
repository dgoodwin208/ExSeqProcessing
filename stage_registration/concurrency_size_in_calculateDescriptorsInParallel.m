function max_run_jobs = concurrency_size_in_calculateDescriptorsInParallel(cond)

    loadParameters;

    if ~isfield(params,'CALC_DESC_MAX_RUN_JOBS')
        if params.USE_GPU_CUDA
            job_worker_mem_usage = 10 * cond.downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        else
            job_worker_mem_usage = 12 * cond.downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        end
        max_run_jobs = min(params.NUM_ROUNDS,floor(cond.availablemem / job_worker_mem_usage));
        expected_total_mem_usage = job_worker_mem_usage * max_run_jobs;

        fprintf('## expected total memory usage = %7.1f MiB\n',expected_total_mem_usage);
        fprintf('## expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
    else
        max_run_jobs = params.CALC_DESC_MAX_RUN_JOBS;
    end

    fprintf('## CALC_DESC_MAX_RUN_JOBS = %d\n',max_run_jobs);
end
