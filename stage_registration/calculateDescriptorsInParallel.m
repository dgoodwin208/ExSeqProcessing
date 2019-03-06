function success_code = calculateDescriptorsInParallel()

    loadParameters;

    run_num_list = 1:params.NUM_ROUNDS;

    arg_list = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    max_run_jobs = concurrency_size_in_calculateDescriptorsInParallel();
    max_pool_size = 0;

    num_jobs = length(run_num_list);

    [success_code, output] = batch_process('reg1-calcDesc', @calculateDescriptors, run_num_list, arg_list, ...
        postfix_list, max_pool_size, num_jobs, max_run_jobs, params.WAIT_SEC, params.logDir);

end

