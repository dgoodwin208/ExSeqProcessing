function success_code = registerWithCorrespondencesInParallel()

    loadParameters;

    run_num_list = 1:params.NUM_ROUNDS;
    run_num_list(regparams.FIXED_RUN) = [];

    arg_list_1 = {};
    arg_list_2 = {};
    arg_list_3 = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list_1{end+1} = {run_num};
        arg_list_2{end+1} = {run_num,false};
        arg_list_3{end+1} = {run_num,true};
        postfix_list{end+1} = num2str(run_num);
    end

    max_jobs = length(run_num_list);

    disp('===== calc-correspondences');
    [success_code, output] = batch_process('reg2-calcCorr', @calcCorrespondences, run_num_list, arg_list_1, ...
        postfix_list, params.REG_CORR_MAX_POOL_SIZE, max_jobs, params.REG_CORR_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.REG_CORR_MAX_RUN_JOBS');
        disp('params.REG_CORR_MAX_POOL_SIZE');
        disp('params.REG_CORR_MAX_THREADS');
        return;
    end

    disp('===== register-with-correspondences');
    [success_code, output] = batch_process('reg2-regWCorr', @registerWithCorrespondences, run_num_list, arg_list_2, ...
        postfix_list, params.REG_CORR_MAX_POOL_SIZE, max_jobs, params.REG_CORR_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.REG_CORR_MAX_RUN_JOBS');
        disp('params.REG_CORR_MAX_POOL_SIZE');
        disp('params.REG_CORR_MAX_THREADS');
        return;
    end

    disp('===== register-with-downsampled-correspondences');
    [success_code, output] = batch_process('reg2-regWDownSampleCorr', @registerWithCorrespondences, run_num_list, arg_list_3, ...
        postfix_list, params.REG_CORR_MAX_POOL_SIZE, max_jobs, params.REG_CORR_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.REG_CORR_MAX_RUN_JOBS');
        disp('params.REG_CORR_MAX_POOL_SIZE');
        disp('params.REG_CORR_MAX_THREADS');
        return;
    end

end
