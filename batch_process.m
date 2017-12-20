function batch_processing(prefix, func, run_num_list, args)

    loadExperimentParams;

    disp('set up cluster')
    tic;
    cluster = parcluster('local_96workers');
    %parpool(cluster,24)
    toc;

    tic; 
    disp('===== create batch jobs =====') 
    max_running_jobs = params.JOB_SIZE;
    max_jobs = length(run_num_list);
    if func == @calculateDescriptors
        run_num_list_size = length(run_num_list);
        desc_size = params.ROWS_DESC * params.COLS_DESC;
        max_jobs  = run_num_list_size * desc_size;
    end
    waiting_sec = 10;

    jobs = cell(1, max_jobs);
    running_jobs = zeros(1, max_jobs);
    job_idx = 1;

    while job_idx <= max_jobs || sum(running_jobs) > 0
        % check that number of jobs currently running is valid
        if (job_idx <= max_jobs) && (sum(running_jobs) < max_running_jobs)

            % determine args
            run_num = run_num_list(job_idx);
            if func == @registerWithDescriptors
                args = {run_num};
            else if func == @calculateDescriptors
                [run_num, target_idx] = getJobIds(run_num_list, job_idx, desc_size);
                args = {run_num, target_idx, target_idx};
            else if func == @normalizeImage
                % FIXME add one more element to cell
                args = {args, run_num};
            end

            disp(['create batch ', num2str(job_idx)]);
            running_jobs(job_idx) = 1; % mark as running
            jobs{job_idx} = batch(cluster, func, ... 
                0, args, 'Pool', 2, 'CaptureDiary', true);
            job_idx = job_idx + 1;
        else
            for job_idx_running = find(running_jobs==1)
                job = jobs{job_idx_running};
                is_finished = 0;

                % determine args
                run_num = run_num_list(job_idx_running);
                postfix = num2str(run_num);
                if func == @registerWithDescriptors
                    args = {run_num};
                else if func == @calculateDescriptors
                    [run_num, target_idx] = getJobIds(run_num_list, job_idx_running, desc_size);
                    args = {run_num, target_idx, target_idx};
                    postfix = [num2str(run_num), '-', num2str(target_idx)];
                else if func == @normalizeImage
                    % FIXME add one more element to cell
                    args = {args, run_num};
                end

                if strcmp(job.State,'finished')
                    if isempty(getReport(job.Tasks(1).Error))
                        % batch finished with no error
                        disp(['batch (',num2str(job_idx_running),') has ', job.State,'.']);
                        diary(job, ['./matlab-', prefix, '-', postfix, '.log']);
                    else
                        %batch finished with internal error
                        disp(['batch (',num2str(job_idx_running),') had fatal internal error, see log file.']);
                        fn = strcat('./matlab-', prefix, '-', postfix, '-fatal.log');
                        diary(job, fn);
                        % guarantee error message gets logged (appended)
                        error_msg = getReport(job.Tasks(1).Error)
                        fid = fopen(fn, 'a');
                        fprintf(fid, error_msg);
                        fclose(fid);
                    end
                    running_jobs(job_idx_running) = 0;
                    delete(job);
                    is_finished = 1;
                elseif strcmp(job.State, 'failed') 
                    % batch call failed, retry
                    disp(['batch (',num2str(job_idx_running),') has ',job.State,', resubmit it.']);
                    diary(job, ['./matlab-', prefix, '-', postfix, '-failed.log']);
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

