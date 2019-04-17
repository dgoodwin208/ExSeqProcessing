function success_code = registerWithCorrespondencesCUDAInParallel()

    loadParameters;

    calcCorrespondencesForFixed(params.REFERENCE_ROUND_WARP);

    run_num_list = 1:params.NUM_ROUNDS;
    run_num_list(params.REFERENCE_ROUND_WARP) = [];

    arg_list = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    run_num_list_orig_and_downsample = [];
    arg_list_orig_and_downsample = {};
    postfix_list_orig_and_downsample = {};
    for run_num = run_num_list
        run_num_list_orig_and_downsample(end+1) = run_num;
        run_num_list_orig_and_downsample(end+1) = run_num;
        arg_list_orig_and_downsample{end+1} = {run_num,true};
        arg_list_orig_and_downsample{end+1} = {run_num,false};
        postfix_list_orig_and_downsample{end+1} = num2str(run_num);
        postfix_list_orig_and_downsample{end+1} = num2str(run_num);
    end

    num_jobs = length(run_num_list);

    conditions = conditions_for_concurrency();
    [calc_corr_max_run_jobs,affine_max_run_jobs,affine_max_pool_size] = concurrency_size_in_registerWithCorrespondencesCUDAInParallel(conditions);
    calc_corr_max_pool_size = 0;

    disp('===== calc-correspondences-in-cuda');
    [success_code, output] = batch_process('reg2-calcCorrCuda', @calcCorrespondencesCUDA, run_num_list, arg_list, ...
        postfix_list, calc_corr_max_pool_size, num_jobs, calc_corr_max_run_jobs, params.WAIT_SEC, params.logDir);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.CALC_CORR_MAX_RUN_JOBS');
        disp('params.CALC_CORR_MAX_THREADS');
        return;
    end

    num_jobs_orig_and_downsample = length(run_num_list_orig_and_downsample);

    disp('===== perform-affine-transforms');
    [success_code, output] = batch_process('reg2-affine', @performAffineTransforms, run_num_list_orig_and_downsample, arg_list_orig_and_downsample, ...
        postfix_list_orig_and_downsample, affine_max_pool_size, num_jobs_orig_and_downsample, affine_max_run_jobs, params.WAIT_SEC, params.logDir);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.AFFINE_MAX_RUN_JOBS');
        disp('params.AFFINE_MAX_POOL_SIZE');
        disp('params.AFFINE_MAX_THREADS');
        return;
    end

    if strcmp(regparams.REGISTRATION_TYPE,'affine')
        fprintf('Ending the registration after the affine\n');
        return;
    end

    conditions = conditions_for_concurrency();
    [tps3dwarp_max_run_jobs,tps3dwarp_max_pool_size] = concurrency_size_in_TPS3DWarping(conditions);

    disp('===== TPS3D-warping');
    [success_code, output] = batch_process('reg2-TPS3DWarp', @TPS3DWarping, run_num_list_orig_and_downsample, arg_list_orig_and_downsample, ...
        postfix_list_orig_and_downsample, tps3dwarp_max_pool_size, num_jobs_orig_and_downsample, tps3dwarp_max_run_jobs, params.WAIT_SEC, params.logDir);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.TPS3DWARP_MAX_RUN_JOBS');
        disp('params.TPS3DWARP_MAX_POOL_SIZE');
        disp('params.TPS3DWARP_MAX_THREADS');
        return;
    end

end
