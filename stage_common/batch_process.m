function [success_code, outputs] = batch_process(prefix, func, run_num_list, arg_list, ...
    postfix_list, pool, max_jobs, max_running_jobs, wait_sec, output_num, channels)

    success_code = true;
    outputs = {};

    disp('set up cluster')
    tic;
    cluster = parcluster('local_logical_cores');
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
                postfix = postfix_list{job_idx_running};

                % retry out of memory errors
                nomem_error = 0;
                msg = 0;
                if ~isempty(job.Tasks(1)) && ~isempty(job.Tasks(1).Error) &&...
                    ~isempty(job.Tasks(1).Error.identifier)
                    msg = 1;
                    nomem_error = strcmp(job.Tasks(1).Error.identifier, 'MATLAB:nomem');
                end
                if strcmp(job.State,'finished') && ~nomem_error
                    %if isempty(job.Tasks(1).Error) && ~isequal(class(job.Tasks(1).Error), 'ParallelException')
                    if isempty(job.Tasks(1).Error)
                        % batch finished with no error
                        disp(['batch (',num2str(job_idx_running),') has ', job.State,'.']);
                        diary(job, ['./matlab-', prefix, '-', postfix, '.log']);
                        if output_num
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
                elseif strcmp(job.State, 'failed') || nomem_error
                    % batch call failed, retry
                    disp(['batch (',num2str(job_idx_running),') has failed, resubmitting...']);
                    if msg
                        fprintf('Error caused by:\n%s\n', job.Tasks(1).Error.message);
                    end
                    diary(job, ['./matlab-', prefix, '-', postfix, '-failed.log']);
                    %jobs{job_idx_running} = recreate(job); % causes misc. errors
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

