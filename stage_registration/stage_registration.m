function stage_registration()

    loadParameters;

    % sub-stage 1: calculate descriptors
    ret = make_links_in_normalization_dir();
    if ~ret
        return
    end

    [ret,messages] = check_files_in_calculateDescriptors();
    if ~ret
        if params.USE_GPU_CUDA
            calculateDescriptorsCUDAInParallel();
        else
            calculateDescriptorsInParallel();
        end

        [ret,messages] = check_files_in_calculateDescriptors();
        if ~ret
            for i = 1:length(messages)
                disp(messages{i})
            end
            return
        end
    else
        fprintf('already processed calculateDescriptors\n');
    end

    % sub-stage 2: register with correspondences
    ret = make_links_in_registration_dir();
    if ~ret
        return
    end

    [ret,messages] = check_files_in_registerWithCorrespondences();
    if ret
        fprintf('already processed registerWithCorrespondences\n');
        fprintf('[DONE]\n');
        return
    end

    if params.USE_GPU_CUDA
        registerWithCorrespondencesCUDAInParallel();
    else
        registerWithCorrespondencesInParallel();
    end

    [ret,messages] = check_files_in_registerWithCorrespondences();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end

