% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = calculateDescriptorsInParallel(run_num_list)

    % get max_jobs
    loadParameters;
    sem_name = sprintf('/%s.gc',getenv('USER'));

    semaphore(sem_name,'open',1);
    ret = semaphore(sem_name,'getvalue');
    if ret ~= 1
        semaphore(sem_name,'unlink');
        semaphore(sem_name,'open',1);
    end

    run_num_list_size = length(run_num_list);
    desc_size = regparams.ROWS_DESC * regparams.COLS_DESC;
    max_jobs  = run_num_list_size * desc_size;
    cuda=true;

    arg_list = {};
    postfix_list = {};
    for job_idx = 1:max_jobs
        [run_num, target_idx] = getJobIds(run_num_list, job_idx, desc_size);
        arg_list{end+1} = {run_num, target_idx, target_idx, cuda};
        postfix_list{end+1} = strcat(num2str(run_num), '-', num2str(target_idx));
    end

    [success_code, output] = batch_process('calcDesc', @calculateDescriptors, run_num_list, arg_list, ...
        postfix_list, params.CALC_DESC_MAX_POOL_SIZE, max_jobs, params.CALC_DESC_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

    semaphore(sem_name,'unlink');
end

