% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = registerWithDescriptorsInParallel(run_num_list)

    loadParameters;
    
    arg_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
    end

    max_jobs = length(run_num_list);

    [success_code, output] = batch_process('regDesc', @registerWithDescriptors, run_num_list, arg_list, ...
        params.REG_POOL_SIZE, max_jobs, params.MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

end
