function success_code = calculateDescriptorsInParallel()

    % get max_jobs
    loadParameters;

    run_num_list = 1:params.NUM_ROUNDS;
    run_num_list_size = length(run_num_list);
    desc_size = regparams.ROWS_DESC * regparams.COLS_DESC;
    num_jobs  = run_num_list_size * desc_size;

    arg_list = {};
    postfix_list = {};
    for job_idx = 1:num_jobs
        [run_num, target_idx] = getJobIds(run_num_list, job_idx, desc_size);
        arg_list{end+1} = {run_num, target_idx, target_idx};
        postfix_list{end+1} = strcat(num2str(run_num), '-', num2str(target_idx));
    end

    max_run_jobs = concurrency_size_in_calculateDescriptorsInParallel();
    max_pool_size = 0;

    [success_code, output] = batch_process('reg1-calcDesc', @calculateDescriptors, run_num_list, arg_list, ...
        postfix_list, max_pool_size, num_jobs, max_run_jobs, params.WAIT_SEC, params.logDir);

end

