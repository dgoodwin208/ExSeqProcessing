function varargout = concurrency_size_in_registerWithCorrespondencesInParallel(cond)

    loadParameters;

    num_rounds = params.NUM_ROUNDS - 1;

    if ~isfield(params,'CALC_CORR_MAX_RUN_JOBS') || ~isfield(params,'REG_CORR_MAX_RUN_JOBS')
        % TODO: consider a way to calculate avr # of keys when # of register_channels is over 1
        avr_num_keys = 0;
        for i = 1:num_rounds
            nkeys_filename = fullfile(params.registeredImagesDir,sprintf('nkeys_%s-downsample_round%03d_%s.mat',...
            params.FILE_BASENAME,i,regparams.REGISTERCHANNELS_SIFT{end}));
            load(nkeys_filename,'num_keys');
            avr_num_keys = avr_num_keys + num_keys;
        end
        avr_num_keys = avr_num_keys / num_rounds;

        fprintf('## average # of keypoints = %d\n',avr_num_keys);
    end

    if ~isfield(params,'CALC_CORR_MAX_RUN_JOBS')
        main_mem_usage = params.MATLAB_PROC_CONTEXT;

        size_dbl = 8;
        num_elms_ivec = sift_params.IndexSize^3 * sift_params.nFaces;
        size_ivec_int8 = num_elms_ivec * 1;
        size_ivec_dbl = num_elms_ivec * size_dbl;
        % struct keys
        % 6 (double) + 1 (uint8, ivec)
        size_elem_struct = 176;
        size_struct_keys = 6*(size_dbl+size_elem_struct) + size_ivec_int8+size_elem_struct;
        % 4 # keypoints[3] (double) + 4 # keypoints (double) +
        % 4 # keypoints * ivecs (double) + 3 # keypoints x size(struct keys) + 1.9 # keypoints^2 (double)
        job_worker_mem_usage = (4*3*size_dbl+4*size_dbl+4*size_ivec_dbl+3*size_struct_keys)*avr_num_keys/1024/1024 + ...
            1.9*avr_num_keys^2*size_dbl/1024/1024 + params.MATLAB_PROC_CONTEXT;
        calc_corr_max_run_jobs = min(num_rounds,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        total_mem_usage = main_mem_usage + job_worker_mem_usage*calc_corr_max_run_jobs;

        fprintf('## CALC_CORR: total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## CALC_CORR: expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
    else
        calc_corr_max_run_jobs = params.CALC_CORR_MAX_RUN_JOBS;
    end
    if calc_corr_max_run_jobs == 0
        fprintf('[WARNING] calcCorrespondences func might use too large memory.\n');
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
        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        % for affine transform only
        % 3.8 image size (orig,double) + 5 # keypoints[3] (double) + 1 # keypoints[4] (double)
        % for TPS3DWarp
        % 11 image size (orig,double) + 2 image size (orig,int8) + 5 # keypoints[3] (double) + 1 # keypoints[4] (double)

        size_dbl = 8;
        if strcmp(regparams.REGISTRATION_TYPE,'affine')
            job_worker_mem_usage = 3.8*cond.imgsize_dbl + (5*3+1*4)*size_dbl*avr_num_keys/1024/1024 + params.MATLAB_PROC_CONTEXT;
        else
            job_worker_mem_usage = 11*cond.imgsize_dbl + 2*cond.imgsize_int8 + (5*3+1*4)*size_dbl*avr_num_keys/1024/1024 + ...
                params.MATLAB_PROC_CONTEXT;
        end
        reg_corr_max_run_jobs = min(num_rounds,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        total_mem_usage = main_mem_usage + job_worker_mem_usage*reg_corr_max_run_jobs;

        fprintf('## REG_CORR: total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## REG_CORR: expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
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
