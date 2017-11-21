% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function calculateDescriptorsInParallel(run_num_list)

    loadExperimentParams;
    loadParams;

    run_num_list_size = length(run_num_list);
    desc_size = params.ROWS_DESC * params.COLS_DESC;
    run_size  = run_num_list_size * desc_size;

    disp('set up cluster')
    tic;
    cluster = parcluster('local_96workers');
    toc; 

    tic; 
    disp('===== create batch jobs =====') 
    max_running_jobs = params.JOB_SIZE;
    max_jobs = run_size;
    waiting_sec = 10;

    jobs = cell(1, max_jobs);
    running_jobs = zeros(1, max_jobs);
    job_idx = 1;

    while job_idx <= max_jobs || sum(running_jobs) > 0
        % check that number of jobs currently running is valid
        if (job_idx <= max_jobs) && (sum(running_jobs) < max_running_jobs)
            [run_num, target_idx] = getJobIds(run_num_list, job_idx, desc_size);

            disp(['create batch (', num2str(job_idx), ') run_num=', ...
                num2str(run_num),', target_idx=',num2str(target_idx)]);
            running_jobs(job_idx) = 1; % mark as running
            jobs{job_idx} = batch(cluster, @calculateDescriptors, ... 
                0, {run_num, target_idx, target_idx}, 'Pool', 2, 'CaptureDiary', true);
            job_idx = job_idx + 1;
        else
            for job_idx_running = find(running_jobs==1)
                job = jobs{job_idx_running};
                is_finished = 0;

                [run_num, target_idx] = getJobIds(run_num_list, job_idx_running, desc_size);
                if strcmp(job.State,'finished')
                    disp(['batch (',num2str(job_idx_running),') has ', job.State,'.']);
                    diary(job, ['./matlab-calcDesc-',num2str(run_num),'-',num2str(target_idx),'.log']);
                    running_jobs(job_idx_running) = 0;
                    delete(job);
                    is_finished = 1;
                elseif strcmp(job.State,'failed') % Retry block
                    disp(['batch (',num2str(job_idx_running),') has ',job.State,', resubmit it.']);
                    diary(job, ['./matlab-calcDesc-', num2str(run_num), '-', ... 
                        num2str(target_idx),'-failed.log']);
                    jobs{job_idx_running} = batch(cluster, @calculateDescriptors, ... 
                        0, {run_num, target_idx, target_idx}, 'Pool', 2, 'CaptureDiary', true);
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
end

