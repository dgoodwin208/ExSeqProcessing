% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = registerWithDescriptorsInParallel(run_num_list)

    loadParameters;
    loadExperimentParams;

    calculateShapeContextDescriptors(params.FIXED_RUN);

    semaphore('/gr','open',1);
    ret = semaphore('/gr','getvalue');
    if ret ~= 1
        semaphore('/gr','unlink');
        semaphore('/gr','open',1);
    end

    arg_list = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    max_jobs = length(run_num_list);

    [success_code, output] = batch_process('regDesc', @registerWithDescriptors, run_num_list, arg_list, ...
        postfix_list, params.REG_DESC_MAX_POOL_SIZE, max_jobs, params.REG_DESC_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

%    for run_num = run_num_list
%        transformImagesWithAffineAnd3DTPS(run_num);
%    end
    [success_code, output] = batch_process('transformImg', @transformImagesWithAffineAnd3DTPS, run_num_list, arg_list, ...
        postfix_list, params.TRANSF_IMG_MAX_POOL_SIZE, max_jobs, params.TRANSF_IMG_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

    semaphore('/gr','unlink');

end
