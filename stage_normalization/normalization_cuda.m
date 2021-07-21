% normalization

function success_code = normalization_cuda()

    loadParameters;

    arg_list = {};
    postfix_list = {};
    run_num_list = 1:params.NUM_ROUNDS;
    for run_num = run_num_list
        arg_list{end+1} = {params.colorCorrectionImagesDir,params.normalizedImagesDir,params.FILE_BASENAME,params.SHIFT_CHAN_STRS, run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    conditions = conditions_for_concurrency();
    [max_run_jobs,max_run_jobs_downsampled] = concurrency_size_in_normalization_cuda(conditions);
    max_pool_size = 0;

    num_jobs = length(run_num_list);

    [success_code, output] = batch_process('normalization', @normalizeImage_cuda, run_num_list, arg_list, ...
        postfix_list, max_pool_size, num_jobs, max_run_jobs, params.WAIT_SEC, params.logDir);

    if ~params.DO_DOWNSAMPLE
        return
    end

    arg_list_downsample = {};
    for run_num = run_num_list
        arg_list_downsample{end+1} = {params.colorCorrectionImagesDir,params.normalizedImagesDir,[params.FILE_BASENAME,'-downsample'],params.SHIFT_CHAN_STRS, run_num};
    end

    [success_code, output] = batch_process('normalization-downsample', @normalizeImage_cuda, run_num_list, arg_list_downsample, ...
        postfix_list, max_pool_size, num_jobs, max_run_jobs_downsampled, params.WAIT_SEC, params.logDir);

end

