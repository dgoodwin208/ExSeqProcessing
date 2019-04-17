function [calc_corr_max_run_jobs,affine_max_run_jobs,affine_max_pool_size] = concurrency_size_in_registerWithCorrespondencesCUDAInParallel(cond)

    loadParameters;

    num_rounds = params.NUM_ROUNDS - 1;

    if ~isfield(params,'CALC_CORR_MAX_RUN_JOBS') || ~isfield(params,'AFFINE_MAX_RUN_JOBS')
        % TODO: consider a way to calculate avr # of keys when # of register_channels is over 1
        % # of keys is the same between original images and downsampling images
        round_list = 1:params.NUM_ROUNDS;
        round_list(params.REFERENCE_ROUND_WARP) = [];
        avr_num_keys = 0;
        for i = round_list
            nkeys_filename = fullfile(params.registeredImagesDir,sprintf('nkeys_%s-downsample_round%03d_summedNorm.mat',...
            params.FILE_BASENAME,i));
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
        size_struct_keys = 6*(size_dbl+size_elem_struct)+size_ivec_int8+size_elem_struct;
        % 2 # keypoints x struct keys + 6 # keypoints[3] (double) +
        % 3 # keypoints x ivecs (uint8) + 4 # keypoints x ivecs (double) + 1 # keypoints x 1000 (double)
        job_worker_mem_usage = (2*size_struct_keys+6*3*size_dbl+3*size_ivec_int8+4*size_ivec_dbl+1000*size_dbl)*avr_num_keys/1024/1024 + params.MATLAB_PROC_CONTEXT;
        calc_corr_max_run_jobs = min(num_rounds,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        total_mem_usage = main_mem_usage + job_worker_mem_usage*calc_corr_max_run_jobs;

        fprintf('## CALC_CORR: total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## CALC_CORR: expected memory usage / job_worker = %7.1f MiB\n',job_worker_mem_usage);
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
        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        job_worker_mem_usage = params.MATLAB_PROC_CONTEXT;
        par_worker_mem_usage = 2*cond.imgsize_dbl + params.MATLAB_PROC_CONTEXT;

        job_mem_usage = job_worker_mem_usage + par_worker_mem_usage*affine_max_pool_size;
        affine_max_run_jobs = min(2*num_rounds,floor((cond.availablemem - main_mem_usage) / job_mem_usage));
        if affine_max_run_jobs < 1
            affine_max_pool_size = 1;
            job_mem_usage = job_worker_mem_usage + par_worker_mem_usage*affine_max_pool_size;
            affine_max_run_jobs = min(2*num_rounds,floor((cond.availablemem - main_mem_usage) / job_mem_usage));
        end
        total_mem_usage = main_mem_usage + job_mem_usage*affine_max_run_jobs;

        fprintf('## AFFINE: total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## AFFINE: expected memory usage / job = %7.1f MiB\n',job_mem_usage);
        fprintf('## AFFINE: expected memory usage / par_worker = %7.1f MiB\n',par_worker_mem_usage);
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
end
