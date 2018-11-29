% color correction

function success_code = colorcorrection_3D_cuda()

    loadParameters;

    num_sem_gpus = ones(1, gpuDeviceCount());
    quantilenorm_cuda_init(num_sem_gpus);

    arg_list = {};
    postfix_list = {};
    run_num_list = 1:params.NUM_ROUNDS;
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    max_jobs = length(run_num_list);

    [success_code, output] = batch_process('color-correction', @colorcorrection_3D_poc, run_num_list, arg_list, ...
        postfix_list, params.COLOR_CORRECTION_MAX_POOL_SIZE, max_jobs, params.COLOR_CORRECTION_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);

    quantilenorm_cuda_final(length(num_sem_gpus));
end

