function [max_run_jobs,max_run_jobs_downsampled] = concurrency_size_in_normalization()

    loadParameters;

    if ~isfield(params,'NORM_MAX_RUN_JOBS')
        [imgsize_dbl,dim] = imagesize_in_double(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.tif',params.FILE_BASENAME,params.CHAN_STRS{1})));
        downsample_imgsize_dbl = imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
        max_availablemem = availablememory();
        availablemem = max_availablemem * params.USABLE_MEM_RATE;
        expected_mem_usage = 6 * 4 * imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_run_jobs = min(params.NUM_ROUNDS,uint32(availablemem / expected_mem_usage));

        fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',max_availablemem,availablemem);
        fprintf('## image size (double) = %6.1f MiB, downsampling image size (double) = %6.1f MiB\n',imgsize_dbl,downsample_imgsize_dbl);
        fprintf('## expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
    else
        max_run_jobs = params.NORM_MAX_RUN_JOBS;
    end

    fprintf('## NORM_MAX_RUN_JOBS = %d\n',max_run_jobs);

    if ~isfield(params,'NORM_DOWNSAMPLE_MAX_RUN_JOBS')
        if ~exist('imgsize_dbl','var')
            [imgsize_dbl,dim] = imagesize_in_double(fullfile(params.deconvolutionImagesDir,sprintf('%s_round001_%s.tif',params.FILE_BASENAME,params.CHAN_STRS{1})));
            downsample_imgsize_dbl = imgsize_dbl / (params.DOWNSAMPLE_RATE^3);
        end
        max_availablemem = availablememory();
        availablemem = max_availablemem * params.USABLE_MEM_RATE;
        expected_mem_usage = 6 * 4 * downsample_imgsize_dbl + params.MATLAB_PROC_CONTEXT;
        max_run_jobs_downsampled = min(params.NUM_ROUNDS,uint32(availablemem / expected_mem_usage));

        fprintf('## max available memory = %7.1f MiB, available memory = %7.1f MiB\n',max_availablemem,availablemem);
        fprintf('## image size (double) = %6.1f MiB, downsampling image size (double) = %6.1f MiB\n',imgsize_dbl,downsample_imgsize_dbl);
        fprintf('## expected memory usage / job = %7.1f MiB\n',expected_mem_usage);
    else
        max_run_jobs_downsampled = params.NORM_DOWNSAMPLE_MAX_RUN_JOBS;
    end

    fprintf('## NORM_DOWNSAMPLE_MAX_RUN_JOBS = %d\n',max_run_jobs_downsampled);
end
