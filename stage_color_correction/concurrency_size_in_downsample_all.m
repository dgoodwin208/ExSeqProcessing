function max_pool_size = concurrency_size_in_downsample_all(cond)

    loadParameters;

    if ~isfield(params,'DOWN_SAMPLING_MAX_POOL_SIZE')
        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        par_worker_mem_usage = 2 * cond.imgsize_dbl + cond.downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_pool_size = min(params.NUM_ROUNDS,floor((cond.availablemem - main_mem_usage) / par_worker_mem_usage));

        total_mem_usage = main_mem_usage + par_worker_mem_usage*max_pool_size;

        fprintf('## total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## expected memory usage / par_worker = %7.1f MiB\n',par_worker_mem_usage);
    else
        max_pool_size = params.DOWN_SAMPLING_MAX_POOL_SIZE;
    end

    fprintf('## DOWN_SAMPLING_MAX_POOL_SIZE = %d\n',max_pool_size);
end
