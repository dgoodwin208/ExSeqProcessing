% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function calculateDescriptorsInParallel(run_num_list)

    loadExperimentParams;

    run_num_list_size = length(run_num_list);
    desc_size = params.ROWS_DESC * params.COLS_DESC;
    run_size  = run_num_list_size * desc_size;

    disp('set up cluster')
    tic;
    cluster = parcluster('local_96workers');
    toc; 

    tic; 
    disp('===== create batch jobs =====') 
    max_running_jobs = params.MAX_JOB_RUNNING_NUM;
    max_jobs = params.MAX_JOB_NUM;
    waiting_sec = 10;

    jobs = cell(1, max_jobs);
    running_jobs = zeros(1, max_jobs);
    job_idx = 1;

    while job_idx <= max_jobs || sum(running_jobs) > 0
        % check that number of jobs currently running is valid
        if (job_idx <= max_running_jobs) && (sum(running_jobs) < max_running_jobs)
            % determine run_number idx 
            run_num  = run_num_list(ceil(job_idx / desc_size));

            % convert job_idx 1 - 9 t
            target_idx = mod(job_idx, desc_size);
            if target_idx == 0
                target_idx = desc_size;
            end

            disp(['create batch (',num2str(job_idx),') run_num=',num2str(run_num),', target_idx=',num2str(target_idx)])
            running_jobs(job_idx) = 1; % mark as running
            jobs{job_idx} = batch(cluster, @calculateDescriptors, ... 
                0, {run_num, target_idx, target_idx}, 'Pool', 2, 'CaptureDiary', true);
            job_idx = job_idx + 1;
        else
            for job_id = find(running_jobs==1)
                job = jobs{job_id};
                is_finished = 0;

                % determine run_number idx 
                run_num  = run_num_list(ceil(job_idx / desc_size));

                % convert job_idx 1 - 9 t
                target_idx = mod(job_idx, desc_size);
                if target_idx == 0
                    target_idx = desc_size;
                end
                if strcmp(job.State,'finished')
                    disp(['batch (',num2str(job_id),') has ',job.State,'.'])
                    diary(job, ['./matlab-calcDesc-',num2str(run_num),'-',num2str(target_idx),'.log']);
                    running_jobs(job_id) = 0;
                    delete(job)
                    is_finished = 1;
                elseif strcmp(job.State,'failed')
                    disp(['batch (',num2str(job_id),') has ',job.State,', resubmit it.'])
                    diary(job, ['./matlab-calcDesc-', num2str(run_num), '-', ... 
                        num2str(target_idx),'-failed.log']);
                    % FIXME the retry to correct function
                    jobs{job_id} = batch(cluster,@normalizeImage,0, ...
                       {src_folder_name,dst_folder_name,fileroot_name,channels,job_id}, ...
                       'CaptureDiary',true);
                end
            end
            if is_finished == 0
              disp(['waiting... # of jobs = ',num2str(length(find(running_jobs==1))),', ',num2str(find(running_jobs==1))])
              pause(waiting_sec);
            end
        end
    end

    %jobs = cell(1, run_size);
    %for i = 1:run_size
        %run_num  = run_num_list(ceil(i / desc_size));
        %target_idx = mod(i, desc_size);
        %if target_idx == 0
            %target_idx = desc_size;
        %end
        %disp(['create batch (',num2str(i),') run_num=',num2str(run_num),', target_idx=',num2str(target_idx)])
        %jobs{i} = batch(cluster,@calculateDescriptors,0,{run_num,target_idx,target_idx},'Pool',2,'CaptureDiary',true);
    %end
    %toc;

    %tic;
    %disp('waiting batch jobs...')
    %for i = 1:run_size
        %run_num    = run_num_list(ceil(i / desc_size));
        %target_idx = mod(i, desc_size);
        %if target_idx == 0
            %target_idx = desc_size;
        %end
        %wait(jobs{i})
        %diary(jobs{i},['./matlab-calcDesc-',num2str(run_num),'-',num2str(target_idx),'.log']);
    %end

    %disp('all batch jobs finished')
    %toc;

    %tic;
    %disp('delete batch jobs')
    %for i = 1:run_size
        %delete(jobs{i})
    %end
    %toc;


end

