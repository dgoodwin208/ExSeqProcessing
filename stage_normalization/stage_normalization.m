function stage_normalization()

    loadParameters;
    %Save the loadParameters into the output director in case we need to do
    %later debugging
    copyfile('loadParameters.m',fullfile(params.normalizedImagesDir,...
        sprintf('loadParameters.m.log-%s',date)))
    [ret,messages] = check_files_in_normalization();
    if ret
        fprintf('already processed normalization\n');
        fprintf('[DONE]\n');
        return
    end

    if strcmp(params.NORMALIZE_METHOD,'quantile') && params.USE_GPU_CUDA
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

