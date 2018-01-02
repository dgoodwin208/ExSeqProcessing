function success_code = punctafeinder()

    loadParameters;

    run_num_list = 1:params.NUM_ROUNDS;
    arg_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
    end

    %centroids are the location
    [success_code,centroids] = batch_process('puncta-extraction', @punctafeinder_round, run_num_list, arg_list,...
        params.PUNCTA_POOL_SIZE, params.NUM_ROUNDS, params.PUNCTA_MAX_RUN_JOBS, params.WAIT_SEC, 1, params.NUM_CHANNELS);

    %total_round_num = params.NUM_ROUNDS;
    %jobs = cell(1,total_round_num);
    %running_jobs = zeros(1,total_round_num);
    %roundnum = 1;


    %while roundnum <= total_round_num || sum(running_jobs) > 0
        %if (roundnum <= total_round_num) && (sum(running_jobs) < max_running_jobs)
            %fprintf('create batch (%d)\n',roundnum)
            %if in_parallel
                %running_jobs(roundnum) = 1;
                %jobs{roundnum} = batch(cluster,@punctafeinder_round,1,{roundnum},'Pool',4,'CaptureDiary',true);
            %else
                %centroids_round = punctafeinder_round(roundnum);
                %for c_idx = 1:params.NUM_CHANNELS
                    %centroids{roundnum,c_idx} = centroids_round{roundnum,c_idx};
                %end
            %end
            %roundnum = roundnum+1;
        %else
            %for job_id = find(running_jobs==1)
                %job = jobs{job_id};
                %is_finished = 0;
                %if strcmp(job.State,'finished')
                    %fprintf('batch (%d) has %s.\n',job_id,job.State);
                    %output = fetchOutputs(job);
                    %centroids_job = output{1};
                    %for c_idx = 1:params.NUM_CHANNELS
                        %centroids{job_id,c_idx} = centroids_job{job_id,c_idx};
                    %end
                    %diary(job,['./matlab-puncta-extraction-',num2str(job_id),'.log']);
                    %running_jobs(job_id) = 0;
                    %delete(job)
                    %is_finished = 1;
                %elseif strcmp(job.State,'failed')
                    %fprintf('batch (%d) has %s, resubmit it.\n',job_id,job.State);
                    %diary(job,['./matlab-puncta-extraction-',num2str(job_id),'-failed.log']);
                    %jobs{job_id} = recreate(job);
                %end
            %end
            %if is_finished == 0
              %disp(['waiting... # of jobs = ',num2str(length(find(running_jobs==1))),'; ',num2str(find(running_jobs==1))])
              %pause(waiting_sec);
            %end
        %end
    %end

    %disp('===== all batch jobs finished')
    %toc;

    makeCentroidsAndVoxels(centroids);

end
