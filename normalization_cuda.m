% normalization

function success_code = normalization_cuda(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num)

    loadParameters;

    if length(channels) ~= 4
        disp('# of channels is not 4.')
        return
    end

%    num_sem_gpus = [1,1];
    num_sem_gpus = ones(1, gpuDeviceCount());
%    num_sem_cores = [20, 10, 15, 1];
    num_sem_cores = [params.NORM_MAX_RUN_JOBS, 0, 20, 0];
    quantilenorm_cuda_init(num_sem_gpus,num_sem_cores);

    arg_list = {};
    run_num_list = 1:total_round_num;
    for run_num = run_num_list
        arg_list{end+1} = {src_folder_name,dst_folder_name,fileroot_name,channels, run_num};
    end

    max_jobs = length(run_num_list);

    [success_code, output] = batch_process('normalization', @normalizeImage, run_num_list, arg_list, ...
        params.NORM_EACH_JOB_POOL_SIZE, max_jobs, params.NORM_MAX_RUN_JOBS, params.WAIT_SEC, 0, []);

%    quantilenorm_final(length(num_cores));

end


