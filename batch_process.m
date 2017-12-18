function batch_processing(prefix, func, args, run_num_list)

    loadExperimentParams;

    disp('set up cluster')
    tic;
    cluster = parcluster('local_96workers');
    %parpool(cluster,24)
    toc;

    tic; 
    disp('===== create batch jobs =====') 
    max_jobs = length(run_num_list);
    max_running_jobs = params.JOB_SIZE;
    waiting_sec = 10;

    jobs = cell(1, max_jobs);
    running_jobs = zeros(1, max_jobs);
    job_idx = 1;

    while job_idx <= max_jobs || sum(running_jobs) > 0
        % check that number of jobs currently running is valid
        if (job_idx <= max_jobs) && (sum(running_jobs) < max_running_jobs)
            run_num = run_num_list(job_idx);
            disp(['create batch ', num2str(job_idx)]);
            running_jobs(job_idx) = 1; % mark as running
            jobs{job_idx} = batch(cluster, func, ... 
                0, args, 'Pool', 2, 'CaptureDiary', true);
            job_idx = job_idx + 1;
        else
            for job_idx_running = find(running_jobs==1)
                job = jobs{job_idx_running};
                run_num = run_num_list(job_idx_running);
                is_finished = 0;

                if strcmp(job.State,'finished')
                    if isempty(getReport(job.Tasks(1).Error))
                        disp(['batch (',num2str(job_idx_running),') has ', job.State,'.']);
                        diary(job, ['./matlab-', prefix, '-', num2str(run_num), '.log']);
                    else
                        disp(['batch (',num2str(job_idx_running),') had fatal internal error, see log file.']);
                        fn = strcat('./matlab-', prefix, '-', num2str(run_num), '-fatal.log');
                        diary(job, fn);
                        % guarantee error message gets logged
                        error_msg = getReport(job.Tasks(1).Error)
                        fid = fopen(fn, 'a');
                        fprintf(fid, error_msg);
                        fclose(fid);
                    end
                    running_jobs(job_idx_running) = 0;
                    delete(job);
                    is_finished = 1;
                elseif strcmp(job.State, 'failed') % Retry block
                    disp(['batch (',num2str(job_idx_running),') has ',job.State,', resubmit it.']);
                    diary(job, ['./matlab-', prefix, '-', num2str(run_num), '-failed.log']);
                    jobs{job_idx_running} = batch(cluster, func, ... 
                        0, args, 'Pool', 2, 'CaptureDiary', true);
                end
            end
            if ~is_finished
              disp(['waiting on ', num2str(length(find(running_jobs==1))), ...
                  ' jobs (batches : ', num2str(find(running_jobs==1)), ')']);
              pause(waiting_sec);
            end
        end
    end
    toc;

    disp('delete parpool')
    tic;
    p = gcp('nocreate');
    delete(p);
    toc;
end

