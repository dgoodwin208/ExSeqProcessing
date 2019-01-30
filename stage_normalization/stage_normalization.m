function stage_normalization()

    loadParameters;

    [ret,messages] = check_files_in_normalization();
    if ret
        fprintf('already processed normalization\n');
        fprintf('[DONE]\n');
        return
    end

    if params.USE_GPU_CUDA
        normalization_cuda();
    else
        normalization();
    end

    [ret,messages] = check_files_in_normalization();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end

