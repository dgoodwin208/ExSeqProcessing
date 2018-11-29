% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = batch_memory_errors()

    % get max_jobs
    loadExperimentParams;

    run_num_list = [1]
    run_num_list_size = length(run_num_list);
    desc_size = params.ROWS_DESC * params.COLS_DESC;
    max_jobs  = run_num_list_size * desc_size;

    arg_list = {};
    postfix_list = {};
    for job_idx = 1:max_jobs
        arg_list{end+1} = {};
        postfix_list{end+1} = num2str(job_idx);
    end

    loadParameters;
    [success_code, output] = batch_process('createMemError', @create_memory_error, run_num_list, arg_list, ...
        postfix_list, params.CALC_DESC_MAX_POOL_SIZE, max_jobs, params.CALC_DESC_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);

end

