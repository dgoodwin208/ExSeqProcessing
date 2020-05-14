function [max_run_jobs,max_run_jobs_downsampled] = concurrency_size_in_normalization(cond)

    loadParameters;

    if ~isfield(params,'NORM_MAX_RUN_JOBS')
        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        job_worker_mem_usage = 6 * 4 * cond.imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_run_jobs = min(params.NUM_ROUNDS,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        total_mem_usage = main_mem_usage + job_worker_mem_usage*max_run_jobs;

        fprintf('## total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
    else
        max_run_jobs = params.NORM_MAX_RUN_JOBS;
    end

    fprintf('## NORM_MAX_RUN_JOBS = %d\n',max_run_jobs);

    if ~isfield(params,'NORM_DOWNSAMPLE_MAX_RUN_JOBS')
        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        job_worker_mem_usage = 6 * 4 * cond.downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_run_jobs_downsampled = min(params.NUM_ROUNDS,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        total_mem_usage = main_mem_usage + job_worker_mem_usage*max_run_jobs_downsampled;

        fprintf('## total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
    else
        max_run_jobs_downsampled = params.NORM_DOWNSAMPLE_MAX_RUN_JOBS;
    end

    fprintf('## NORM_DOWNSAMPLE_MAX_RUN_JOBS = %d\n',max_run_jobs_downsampled);
end
