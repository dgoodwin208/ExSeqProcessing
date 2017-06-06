% normalization

function normalization(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num)

    if length(channels) ~= 4
        disp('# of channels is not 4.')
        return
    end

    cluster = parcluster('local_96workers');

    tic;
    disp('===== create batch jobs')

    max_running_jobs = 12;
    waiting_sec = 10;

    jobs = cell(1,total_round_num);
    running_jobs = zeros(1,total_round_num);
    roundnum = 1;

    while roundnum <= total_round_num || sum(running_jobs) > 0
        if (roundnum <= total_round_num) && (sum(running_jobs) < max_running_jobs)
            disp(['create batch (',num2str(roundnum),')'])
            running_jobs(roundnum) = 1;
            jobs{roundnum} = batch(cluster,@normalizeImage,0,{src_folder_name,dst_folder_name,fileroot_name,channels,roundnum});
            roundnum = roundnum+1;
        else
            for job_id = find(running_jobs==1)
                job = jobs{job_id};
                is_finished = 0;
                if strcmp(job.State,'finished') || strcmp(job.State,'failed')
                    disp(['batch (',num2str(job_id),') has ',job.State,'.'])
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

function normalizeImage(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum)

    if (exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4}))))
    else
        disp('no channel files.')
        return
    end

    chan1 = load3DTif(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1})));
    chan2 = load3DTif(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2})));
    chan3 = load3DTif(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3})));
    chan4 = load3DTif(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4})));

    data_cols(:,1) = reshape(chan1,[],1);
    data_cols(:,2) = reshape(chan2,[],1);
    data_cols(:,3) = reshape(chan3,[],1);
    data_cols(:,4) = reshape(chan4,[],1);

    %     %Normalize the data
    data_cols_norm = quantilenorm(data_cols);

    % reshape the normed results back into 3d images
    chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
    chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
    chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
    chan4_norm = reshape(data_cols_norm(:,4),size(chan4));


    summed_norm = chan1_norm+chan2_norm+chan3_norm+chan4_norm;

    save3DTif(summed_norm,sprintf('%s/%s_round%03i_summedNorm.tif',dst_folder_name,fileroot_name,roundnum));

end

