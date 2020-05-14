% color correction

function success_code = colorcorrection_3D_cuda()

    loadParameters;

    arg_list = {};
    postfix_list = {};
    run_num_list = 1:params.NUM_ROUNDS;
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    conditions = conditions_for_concurrency();
    max_run_jobs = concurrency_size_in_colorcorrection_3D_cuda(conditions);
    max_pool_size = 0;

    num_jobs = length(run_num_list);

    [success_code, output] = batch_process('color-correction', @colorcorrection_3D_poc, run_num_list, arg_list, ...
        postfix_list, max_pool_size, num_jobs, max_run_jobs, params.WAIT_SEC, params.logDir);
end

