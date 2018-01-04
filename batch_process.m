function [success_code, outputs] = batch_process(prefix, func, run_num_list, arg_list, ...
    pool, max_jobs, max_running_jobs, wait_sec, output_num, channels)

    success_code = true;
    outputs = {};

    disp('set up cluster')
    tic;
    cluster = parcluster('local_96workers');
    toc;

    tic; 
    disp('===== create batch jobs =====') 

    jobs = cell(1, max_jobs);
    running_jobs = zeros(1, max_jobs);
    job_idx = 1;

    while job_idx <= max_jobs || sum(running_jobs) > 0
        % check that number of jobs currently running is valid
        if (job_idx <= max_jobs) && (sum(running_jobs) < max_running_jobs)

            % determine args
            args = arg_list{job_idx};

            disp(['create batch ', num2str(job_idx)]);
            running_jobs(job_idx) = 1; % mark as running
            jobs{job_idx} = batch(cluster, func, ... 
                output_num, args, 'Pool', pool, 'CaptureDiary', true);
            job_idx = job_idx + 1;
        else
            for job_idx_running = find(running_jobs==1)
                job = jobs{job_idx_running};
                is_finished = 0;

                % determine args
                args = arg_list{job_idx_running};
                if isequal(func, @calculateDescriptors)
                    run_num = args{1};
                    target_idx = args{2};
                    postfix = [num2str(run_num), '-', num2str(target_idx)];
                else
                    run_num = run_num_list(job_idx_running);
                    postfix = num2str(run_num);
                end

                if strcmp(job.State,'finished')
                    %if isempty(job.Tasks(1).Error) && ~isequal(class(job.Tasks(1).Error), 'ParallelException')
                    if isempty(job.Tasks(1).Error)
                        % batch finished with no error
                        disp(['batch (',num2str(job_idx_running),') has ', job.State,'.']);
                        diary(job, ['./matlab-', prefix, '-', postfix, '.log']);
                        if isequal(func, @punctafeinder_round) && output_num
                            job_output = fetchOutputs(job);
                            centroids_job = job_output{1};
                            for c_idx = 1:channels
                                outputs{job_idx_running,c_idx} = centroids_job{job_idx_running,c_idx};
                            end
                        end
                    else
                        %batch finished with internal error
                        disp(['batch (',num2str(job_idx_running),') had fatal internal error, see log file.']);
                        fn = strcat('./matlab-', prefix, '-', postfix, '-fatal.log');
                        diary(job, fn);
                        % guarantee error message gets logged (appended)
                        if ~isempty(job.Tasks(1).Error)
                            error_msg = getReport(job.Tasks(1).Error)
                            fid = fopen(fn, 'a');
                            fprintf(fid, error_msg);
                            fclose(fid);
                        end
                        success_code = false;
                    end
                    running_jobs(job_idx_running) = 0;
                    delete(job);
                    is_finished = 1;
                elseif strcmp(job.State, 'failed') 
                    % batch call failed, retry
                    disp(['batch (',num2str(job_idx_running),') has ',job.State,', resubmit it.']);
                    diary(job, ['./matlab-', prefix, '-', postfix, '-failed.log']);
                    jobs{job_idx_running} = batch(cluster, func, ... 
                        output_num, args, 'Pool', 2, 'CaptureDiary', true);
                end
            end
            if ~is_finished
              disp(['waiting on ', num2str(length(find(running_jobs==1))), ...
                  ' jobs (batches : ', num2str(find(running_jobs==1)), ')']);
              pause(wait_sec);
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
