function [tps3dwarp_max_run_jobs,tps3dwarp_max_pool_size] = concurrency_size_in_TPS3DWarping(cond)

    loadParameters;

    num_rounds = params.NUM_ROUNDS - 1;

    if ~isfield(params,'TPS3DWARP_MAX_RUN_JOBS')
        round_list = 1:params.NUM_ROUNDS;
        round_list(params.REFERENCE_ROUND_WARP) = [];
        avr_num_affinekeys = 0;
        max_num_affinekeys = 0;
        for i = round_list
            affinekeys_filename = fullfile(params.registeredImagesDir,sprintf('affinekeys_%s_round%03d.h5',params.FILE_BASENAME,i));
            keyM_total = h5read(affinekeys_filename,'/keyM_total');
            avr_num_affinekeys = avr_num_affinekeys + size(keyM_total,1);
            max_num_affinekeys = max(max_num_affinekeys,size(keyM_total,1));
        end
        avr_num_affinekeys = avr_num_affinekeys / num_rounds;

        fprintf('## average # of filtered keypoints = %d\n',avr_num_affinekeys);
        fprintf('## max # of filtered keypoints = %d\n',max_num_affinekeys);

        % 1. in TPS3DWarpWholeInParallel,
        % main proc memory usage:
        % 0 (very small size: KeyM_total, KeyF_total, etc)
        % par worker memory usage:
        % (# filtered keypoints (double)) x (TARGET_CHUNK_SIZE+1)^2 x (ZRES+1)
        %
        % 2. in TPS3DApplyCUDA,
        % job worker proc memory usage:
        % 11 image size (orig,double) + 1 image size (orig,int8)
        TARGET_CHUNK_SIZE = 150; % hard coded in TPS3DWarpWholeInParallel
        ZRES = 5; % hard coded in TPS3DWarpWholeInParallel
        main_mem_usage = params.MATLAB_PROC_CONTEXT;
        par_worker_mem_usage = max_num_affinekeys*8*(TARGET_CHUNK_SIZE+1)^2*(ZRES+1)/1024/1024 + params.MATLAB_PROC_CONTEXT;
        job_worker_mem_usage = 11*cond.imgsize_dbl + 1*cond.imgsize_int8 + params.MATLAB_PROC_CONTEXT;

        tps3dwarp_max_run_jobs = min(2*num_rounds,floor((cond.availablemem - main_mem_usage) / job_worker_mem_usage));

        if ~isfield(params,'TPS3DWARP_MAX_POOL_SIZE')
            limit_pool_size = floor(params.NUM_LOGICAL_CORES / tps3dwarp_max_run_jobs);
            job_available_mem_usage = (cond.availablemem - main_mem_usage) / tps3dwarp_max_run_jobs;
            tps3dwarp_max_pool_size = min(limit_pool_size,floor((job_available_mem_usage - main_mem_usage) / par_worker_mem_usage));
        else
            tps3dwarp_max_pool_size = params.TPS3DWARP_MAX_POOL_SIZE;
        end
        job_mem_usage = max(main_mem_usage + par_worker_mem_usage*tps3dwarp_max_pool_size, job_worker_mem_usage);

        total_mem_usage = main_mem_usage + job_mem_usage*tps3dwarp_max_run_jobs;

        fprintf('## TPS3DWARP: total expected memory usage = %7.1f MiB\n',total_mem_usage);
        fprintf('## TPS3DWARP: expected memory usage / job = %7.1f MiB\n',job_mem_usage);
        fprintf('## TPS3DWARP: expected memory usage / par_worker = %7.1f MiB in TPS3DWarpWholeInParallel\n',par_worker_mem_usage);
        fprintf('## TPS3DWARP: expected memory usage / job_worker = %7.1f MiB in TPS3DApplyCUDA\n',job_worker_mem_usage);
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

end
