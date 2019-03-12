function varargout = concurrency_size_in_registerWithCorrespondencesCUDAInParallel()

    loadParameters;

    num_rounds = params.NUM_ROUNDS - 1;

    if ~isfield(params,'CALC_CORR_MAX_RUN_JOBS') || ~isfield(params,'AFFINE_MAX_RUN_JOBS') || ~isfield(params,'TPS3DWARP_MAX_RUN_JOBS')
        [imgsize_dbl,dim] = imagesize_in_double(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.tif',params.FILE_BASENAME,params.CHAN_STRS{1})));
        downsample_imgsize_dbl = imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
        max_availablemem = availablememory();
        availablemem = max_availablemem * params.USABLE_MEM_RATE;

        % TODO: consider a way to calculate avr # of keys when # of register_channels is over 1
        avr_num_keys = 0;
        for i = 1:num_rounds
            nkeys_filename = fullfile(params.registeredImagesDir,sprintf('nkeys_%s-downsample_round%03d_summedNorm.mat',...
            params.FILE_BASENAME,i));
            load(nkeys_filename,'num_keys');
            avr_num_keys = avr_num_keys + num_keys;
        end
        avr_num_keys = avr_num_keys / num_rounds;

        fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',max_availablemem,availablemem);
        fprintf('## image size (double) = %6.1f MiB, downsampling image size (double) = %6.1f MiB\n',imgsize_dbl,downsample_imgsize_dbl);
    end

    if ~isfield(params,'CALC_CORR_MAX_RUN_JOBS')
        size_dbl = 8;
        num_elms_ivec = sift_params.IndexSize^3 * sift_params.nFaces;
        size_ivec_int8 = num_elms_ivec * 1;
        size_ivec_dbl = num_elms_ivec * size_dbl;
        % struct keys
        % 6 (double) + 1 (uint8, ivec)
        size_elem_struct = 176;
        size_struct_keys = 6*(size_dbl+size_elem_struct) + size_ivec_int8+size_elem_struct;
        % 2 # keypoints x struct keys + 6 # keypoints[3] (double) +
        % 3 # keypoints x ivecs (uint8) + 4 # keypoints x ivecs (double) + 1 # keypoints x 1000 (double)
        expected_mem_usage = (2*size_struct_keys+6*3*size_dbl+3*size_ivec_int8+4*size_ivec_dbl+1000*size_dbl)*avr_num_keys/1024/1024 + params.MATLAB_PROC_CONTEXT;
        calc_corr_max_run_jobs = min(num_rounds,uint32(availablemem / expected_mem_usage));

        fprintf('## CALC_CORR: expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
    else
        calc_corr_max_run_jobs = params.CALC_CORR_MAX_RUN_JOBS;
    end
    if calc_corr_max_run_jobs == 0
        fprintf('[WARNING] calcCorrespondencesCUDA func might use too large memory.\n');
        calc_corr_max_run_jobs = 1;
    end
    fprintf('## CALC_CORR_MAX_RUN_JOBS = %d\n',calc_corr_max_run_jobs);

    if ~isfield(params,'CALC_CORR_MAX_THREADS')
        fprintf('## CALC_CORR_MAX_THREADS = automatic\n');
    else
        fprintf('## CALC_CORR_MAX_THREADS = %d\n',params.CALC_CORR_MAX_THREADS);
    end
    fprintf('##\n');

    if ~isfield(params,'AFFINE_MAX_POOL_SIZE')
        affine_max_pool_size = params.NUM_CHANNELS;
    else
        affine_max_pool_size = params.AFFINE_MAX_POOL_SIZE;
    end
    if ~isfield(params,'AFFINE_MAX_RUN_JOBS')
        % (2 image size (orig,double) + 1 image size (orig,uint16)) x # channels
        imgsize_uint16 = imgsize_dbl / 4;
        expected_mem_usage = params.MATLAB_PROC_CONTEXT + (2*imgsize_dbl+imgsize_uint16+params.MATLAB_PROC_CONTEXT)*affine_max_pool_size;
        affine_max_run_jobs = min(2*num_rounds,uint32(availablemem / expected_mem_usage));
        if affine_max_run_jobs < 1
            affine_max_pool_size = 1;
            expected_mem_usage = 2*imgsize_dbl + params.MATLAB_PROC_CONTEXT;
            affine_max_run_jobs = min(2*num_rounds,uint32(availablemem / expected_mem_usage));
        end

        fprintf('## AFFINE: expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
        fprintf('## AFFINE: expected memory usage / worker = %7.1f MiB\n',expected_mem_usage/affine_max_pool_size);
    else
        affine_max_run_jobs = params.AFFINE_MAX_RUN_JOBS;
    end
    if affine_max_run_jobs == 0
        fprintf('[WARNING] performAffineTransforms func might use too large memory.\n');
        affine_max_run_jobs = 1;
    end
    fprintf('## AFFINE_MAX_RUN_JOBS = %d\n',affine_max_run_jobs);
    fprintf('## AFFINE_MAX_POOL_SIZE = %d\n',affine_max_pool_size);

    if ~isfield(params,'AFFINE_MAX_THREADS')
        fprintf('## AFFINE_MAX_THREADS = automatic\n');
    else
        fprintf('## AFFINE_MAX_THREADS = %d\n',params.AFFINE_MAX_THREADS);
    end
    fprintf('##\n');

    if strcmp(regparams.REGISTRATION_TYPE,'affine')
        varargout{1} = calc_corr_max_run_jobs;
        varargout{2} = affine_max_run_jobs;
        varargout{3} = affine_max_pool_size;
        varargout{4} = 0;
        varargout{5} = 0;
        return;
    end

    if ~isfield(params,'TPS3DWARP_MAX_POOL_SIZE')
        tps3dwarp_max_pool_size = 1;
    else
        tps3dwarp_max_pool_size = params.TPS3DWARP_MAX_POOL_SIZE;
    end
    if ~isfield(params,'TPS3DWARP_MAX_RUN_JOBS')
        % main proc memory usage:
        % 3 image size (orig,double)
        % worker memory usage:
        % 4 image size (orig,double) + 1 image size (orig,int8)
        imgsize_int8 = imgsize_dbl / 8;
        main_mem_usage = 3*imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        worker_mem_usage = 4*imgsize_dbl + 1*imgsize_int8 + params.MATLAB_PROC_CONTEXT;
        expected_mem_usage = main_mem_usage + worker_mem_usage*tps3dwarp_max_pool_size;
        tps3dwarp_max_run_jobs = min(2*num_rounds,uint32(availablemem / expected_mem_usage));

        if ~isfield(params,'TPS3DWARP_MAX_POOL_SIZE') && tps3dwarp_max_run_jobs == 2*num_rounds
            rest_availablemem = availablemem - 2*num_rounds*expected_mem_usage;
            additional_pool_size = rest_availablemem / worker_mem_usage;
            if additional_pool_size > 0
                tps3dwarp_max_pool_size = tps3dwarp_max_pool_size + additional_pool_size;
                expected_mem_usage = main_mem_usage + worker_mem_usage*tps3dwarp_max_pool_size;
                tps3dwarp_max_run_jobs = min(2*num_rounds,uint32(availablemem / expected_mem_usage));
            end
        end

        fprintf('## TPS3DWARP: expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
        fprintf('## TPS3DWARP: expected memory usage / worker = %7.1f MiB\n',expected_mem_usage/tps3dwarp_max_pool_size);
    else
        tps3dwarp_max_run_jobs = params.TPS3DWARP_MAX_RUN_JOBS;
    end
    if tps3dwarp_max_run_jobs == 0
        fprintf('[WARNING] TPS3DWarping func might use too large memory.\n');
        tps3dwarp_max_run_jobs = 1;
    end
    fprintf('## TPS3DWARP_MAX_RUN_JOBS = %d\n',tps3dwarp_max_run_jobs);
    fprintf('## TPS3DWARP_MAX_POOL_SIZE = %d\n',tps3dwarp_max_pool_size);

    if ~isfield(params,'TPS3DWARP_MAX_THREADS')
        fprintf('## TPS3DWARP_MAX_THREADS = automatic\n');
        tps3dwarp_max_threads = 'automatic';
    else
        fprintf('## TPS3DWARP_MAX_THREADS = %d\n',params.TPS3DWARP_MAX_THREADS);
    end
    fprintf('##\n');

    varargout{1} = calc_corr_max_run_jobs;
    varargout{2} = affine_max_run_jobs;
    varargout{3} = affine_max_pool_size;
    varargout{4} = tps3dwarp_max_run_jobs;
    varargout{5} = tps3dwarp_max_pool_size;

end
