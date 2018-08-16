% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = registerWithCorrespondencesCUDAInParallel(run_num_list)

    loadParameters;
    sem_name = sprintf('/%s.gr',getenv('USER'));

    calcCorrespondencesForFixed(regparams.FIXED_RUN);

    semaphore(sem_name,'open',1);
    ret = semaphore(sem_name,'getvalue');
    if ret ~= 1
        semaphore(sem_name,'unlink');
        semaphore(sem_name,'open',1);
    end

    arg_list = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    run_num_list_downsample = [];
    arg_list_downsample = {};
    postfix_list_downsample = {};
    for run_num = run_num_list
        run_num_list_downsample(end+1) = run_num;
        run_num_list_downsample(end+1) = run_num;
        arg_list_downsample{end+1} = {run_num,true};
        arg_list_downsample{end+1} = {run_num,false};
        postfix_list_downsample{end+1} = num2str(run_num);
        postfix_list_downsample{end+1} = num2str(run_num);
    end

    max_jobs = length(run_num_list);

    disp('===== calc-correspondences-in-cuda');
    [success_code, output] = batch_process('regCorr-calcCorrCuda', @calcCorrespondencesCUDA, run_num_list, arg_list, ...
        postfix_list, params.REG_CORR_MAX_POOL_SIZE, max_jobs, params.REG_CORR_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.REG_CORR_MAX_RUN_JOBS');
        disp('params.REG_CORR_MAX_POOL_SIZE');
        disp('params.REG_CORR_MAX_THREADS');
        return;
    end

    max_jobs_downsample = length(run_num_list_downsample);

    disp('===== perform-affine-transforms');
    [success_code, output] = batch_process('regCorr-affine', @performAffineTransforms, run_num_list_downsample, arg_list_downsample, ...
        postfix_list_downsample, params.AFFINE_MAX_POOL_SIZE, max_jobs_downsample, params.AFFINE_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
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
        semaphore(sem_name,'unlink');
        return;
    end

    disp('===== calc-3DTPS-warping');
    [success_code, output] = batch_process('regCorr-calc3DTPSWarp', @calc3DTPSWarping, run_num_list_downsample, arg_list_downsample, ...
        postfix_list_downsample, params.TPS3DWARP_MAX_POOL_SIZE, max_jobs_downsample, params.TPS3DWARP_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.TPS3DWARP_MAX_RUN_JOBS');
        disp('params.TPS3DWARP_MAX_POOL_SIZE');
        disp('params.TPS3DWARP_MAX_THREADS');
        return;
    end

    disp('===== apply-3DTPS-warping');
    [success_code, output] = batch_process('regCorr-apply3DTPSWarp', @apply3DTPSWarping, run_num_list_downsample, arg_list_downsample, ...
        postfix_list_downsample, params.APPLY3DTPS_MAX_POOL_SIZE, max_jobs_downsample, params.APPLY3DTPS_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        disp('when out-of-memory has occurred, please check parameters below in loadParameters.m.');
        disp('params.APPLY3DTPS_MAX_RUN_JOBS');
        disp('params.APPLY3DTPS_MAX_POOL_SIZE');
        disp('params.APPLY3DTPS_MAX_THREADS');
        return;
    end

    semaphore(sem_name,'unlink');

end
