% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = registerWithCorrespondencesInParallel(run_num_list)

    loadParameters;
    
    arg_list_1 = {};
    arg_list_2 = {};
    arg_list_3 = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list_1{end+1} = {run_num};
        arg_list_2{end+1} = {run_num,true};
        arg_list_3{end+1} = {run_num,false};
        postfix_list{end+1} = num2str(run_num);
    end

    max_jobs = length(run_num_list);

    disp('===== calc-correspondeces');
    [success_code, output] = batch_process('regDesc-calcCorr', @calcCorrespondences, run_num_list, arg_list_1, ...
        postfix_list, params.REG_POOL_SIZE, max_jobs, params.MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

    disp('===== register-with-correspondeces');
    [success_code, output] = batch_process('regDesc-regWCorr', @registerWithCorrespondences, run_num_list, arg_list_2, ...
        postfix_list, params.REG_POOL_SIZE, max_jobs, params.MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

    disp('===== register-with-downsampled-correspondeces');
    [success_code, output] = batch_process('regDesc-regWDownSampleCorr', @registerWithCorrespondences, run_num_list, arg_list_3, ...
        postfix_list, params.REG_POOL_SIZE, max_jobs, params.MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

end
