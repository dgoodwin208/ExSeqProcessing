% normalization

function normalization(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num,do_downsample)

    if length(channels) ~= 4
        disp('# of channels is not 4.')
        return
    end
    
    if do_downsample
        fileroot_name = sprintf('%s-%s',fileroot_name, 'downsample');
    end
    cluster = parcluster('local_logical_cores');

    tic;
    disp('===== create batch jobs')

    max_running_jobs = 3;
    waiting_sec = 10;

    jobs = cell(1,total_round_num);
    running_jobs = zeros(1,total_round_num);
    roundnum = 1;

    while roundnum <= total_round_num || sum(running_jobs) > 0
        if (roundnum <= total_round_num) && (sum(running_jobs) < max_running_jobs)
            disp(['create batch (',num2str(roundnum),')'])
            running_jobs(roundnum) = 1;
            jobs{roundnum} = batch(cluster,@normalizeImage,0,{src_folder_name,dst_folder_name,fileroot_name,channels,roundnum},'CaptureDiary',true);
            roundnum = roundnum+1;
        else
            for job_id = find(running_jobs==1)
                job = jobs{job_id};
                is_finished = 0;
                if strcmp(job.State,'finished') || strcmp(job.State,'failed')
                    disp(['batch (',num2str(job_id),') has ',job.State,'.'])
                    diary(job,['./matlab-normalization-',num2str(job_id),'.log']);
                    running_jobs(job_id) = 0;
                    delete(job)
                    is_finished = 1;
                end
            end
            if is_finished == 0
              disp(['waiting... ',num2str(find(running_jobs==1))])
              pause(waiting_sec);
            end
        end
    end

    disp('===== all batch jobs finished')
    toc;

end

