function varargout = concurrency_size_in_registerWithCorrespondencesInParallel()

    loadParameters;

    num_rounds = params.NUM_ROUNDS - 1;

    if ~isfield(params,'CALC_CORR_MAX_RUN_JOBS') || ~isfield(params,'REG_CORR_MAX_RUN_JOBS')
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
        % st_keys: 6 (double) + size(ivec (uint8))
        size_st_keys = 6*size_dbl + size_ivec_int8;
        % 2 image size (downsmpl,double) + 4 # keypoints[3] (double) + 8 # keypoints (double) +
        % 4 # keypoints * ivecs (double) + 5 # keypoints x size(struct keys) + # keypoints^2 (double)
        expected_mem_usage = 2*downsample_imgsize_dbl + (4*3*size_dbl+8*size_dbl+4*size_ivec_dbl+5*size_st_keys)*avr_num_keys/1024/1024 + ...
            avr_num_keys^2*size_dbl/1024/1024 + params.MATLAB_PROC_CONTEXT;
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

    if ~isfield(params,'REG_CORR_MAX_POOL_SIZE')
        reg_corr_max_pool_size = 1;
    else
        reg_corr_max_pool_size = params.REG_CORR_MAX_POOL_SIZE;
    end
    if ~isfield(params,'REG_CORR_MAX_RUN_JOBS')
        % main proc memory usage:
        % 5 image size (orig,double) + 6 # keypoints[3] (double)
        % worker memory usage:
        % 8 image size (orig,double) + 1 image size (orig,int8)
        size_dbl = 8;
        imgsize_int8 = imgsize_dbl / 8;
        main_mem_usage = 5*imgsize_dbl 6*3*size_dbl*avr_num_keys/1024/1024 + params.MATLAB_PROC_CONTEXT;
        worker_mem_usage = 8*imgsize_dbl + 1*imgsize_int8 + params.MATLAB_PROC_CONTEXT;
        expected_mem_usage = main_mem_usage + worker_mem_usage*reg_corr_max_pool_size;
        reg_corr_max_run_jobs = min(num_rounds,uint32(availablemem / expected_mem_usage));

        if ~isfield(params,'REG_CORR_MAX_POOL_SIZE') && reg_corr_max_run_jobs == num_rounds
            rest_availablemem = availablemem - num_rounds*expected_mem_usage;
            additional_pool_size = rest_availablemem / worker_mem_usage;
            if additional_pool_size > 0
                reg_corr_max_pool_size = tps3dwarp_max_pool_size + additional_pool_size;
                expected_mem_usage = main_mem_usage + worker_mem_usage*reg_corr_max_pool_size;
                reg_corr_max_run_jobs = min(num_rounds,uint32(availablemem / expected_mem_usage));
            end
        end

        fprintf('## REG_CORR: expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
        fprintf('## REG_CORR: expected memory usage / worker = %7.1f MiB\n',expected_mem_usage/reg_corr_max_pool_size);
    else
        reg_corr_max_run_jobs = params.REG_CORR_MAX_RUN_JOBS;
    end
    if reg_corr_max_run_jobs == 0
        fprintf('[WARNING] registerWithCorrespondences func might use too large memory.\n');
        reg_corr_max_run_jobs = 1;
    end
    fprintf('## REG_CORR_MAX_RUN_JOBS = %d\n',reg_corr_max_run_jobs);
    fprintf('## REG_CORR_MAX_POOL_SIZE = %d\n',reg_corr_max_pool_size);
    fprintf('##\n');

    varargout{1} = calc_corr_max_run_jobs;
    varargout{2} = reg_corr_max_run_jobs;
    varargout{3} = reg_corr_max_pool_size;

end
