% normalization

function success_code = normalization_cuda()

    loadParameters;

    if length(params.CHAN_STRS) ~= 4
        disp('# of channels is not 4.')
        return
    end

    arg_list = {};
    postfix_list = {};
    run_num_list = 1:params.NUM_ROUNDS;
    for run_num = run_num_list
        arg_list{end+1} = {params.colorCorrectionImagesDir,params.normalizedImagesDir,params.FILE_BASENAME,params.CHAN_STRS, run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    max_jobs = length(run_num_list);

    [success_code, output] = batch_process('normalization', @normalizeImage_cuda, run_num_list, arg_list, ...
        postfix_list, params.NORM_MAX_POOL_SIZE, max_jobs, params.NORM_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);

    if ~params.DO_DOWNSAMPLE
        return
    end

    arg_list_downsample = {};
    for run_num = run_num_list
        arg_list_downsample{end+1} = {params.colorCorrectionImagesDir,params.normalizedImagesDir,[params.FILE_BASENAME,'-downsample'],params.CHAN_STRS, run_num};
    end

    [success_code, output] = batch_process('normalization-downsample', @normalizeImage_cuda, run_num_list, arg_list_downsample, ...
        postfix_list, params.NORM_MAX_POOL_SIZE, max_jobs, params.NORM_MAX_RUN_JOBS, params.WAIT_SEC, params.logDir);

end

