% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = registerWithCorrespondencesCUDAInParallel(run_num_list)

    loadParameters;

    calcCorrespondencesForFixed(regparams.FIXED_RUN);

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
    [success_code, output] = batch_process('regDesc-calcCorrCuda', @calcCorrespondencesCUDA, run_num_list, arg_list, ...
        postfix_list, params.REG_DESC_MAX_POOL_SIZE, max_jobs, params.REG_DESC_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        return;
    end

    max_jobs_downsample = length(run_num_list_downsample);

    disp('===== perform-affine-transforms');
    [success_code, output] = batch_process('regDesc-affine', @performAffineTransforms, run_num_list_downsample, arg_list_downsample, ...
        postfix_list_downsample, params.AFFINE_MAX_POOL_SIZE, max_jobs_downsample, params.AFFINE_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        return;
    end

    if strcmp(regparams.REGISTRATION_TYPE,'affine')
        fprintf('Ending the registration after the affine\n');
        semaphore('/gr','unlink');
        return;
    end

    disp('===== calc-3DTPS-warping');
    [success_code, output] = batch_process('regDesc-calc3DTPSWarp', @calc3DTPSWarping, run_num_list_downsample, arg_list_downsample, ...
        postfix_list_downsample, params.TPS3DWARP_MAX_POOL_SIZE, max_jobs_downsample, params.TPS3DWARP_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        return;
    end

    disp('===== apply-3DTPS-warping');
    [success_code, output] = batch_process('regDesc-apply3DTPSWarp', @apply3DTPSWarping, run_num_list_downsample, arg_list_downsample, ...
        postfix_list_downsample, params.APPLY3DTPS_MAX_POOL_SIZE, max_jobs_downsample, params.APPLY3DTPS_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);
    if ~success_code
        disp('batch job has failed.')
        return;
    end

    semaphore('/gr','unlink');

end
