% normalization

function normalization_small_mem(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num)

    loadParameters;

    if length(channels) ~= 4
        disp('# of channels is not 4.')
        return
    end

    cluster = parcluster('local_200workers');
    %parpool(cluster);

    %num_cores = [30 20];
    num_cores = [10 5];
    %num_cores = [5 5];
    quantilenorm_init(num_cores);

    tic;
    disp('===== create batch jobs')

    max_running_jobs = params.NORM_JOB_SIZE;
    waiting_sec = 10;

    jobs = cell(1,total_round_num);
    running_jobs = zeros(1,total_round_num);
    roundnum = 1;

%    while roundnum <= total_round_num  % serial run
    while roundnum <= total_round_num || sum(running_jobs) > 0
        if (roundnum <= total_round_num) && (sum(running_jobs) < max_running_jobs)
            disp(['create batch (',num2str(roundnum),')'])
            running_jobs(roundnum) = 1;
            jobs{roundnum} = batch(cluster,@normalizeImage,0, ...
               {src_folder_name,dst_folder_name,fileroot_name,channels,roundnum}, ...
               'Pool',params.NORM_EACH_JOB_POOL_SIZE,'CaptureDiary',true);
%            normalizeImage(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum); % serial run
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

    quantilenorm_final(length(num_cores));

end

function normalizeImage(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum)

    loadParameters;

    if (exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4}))))
    else
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1})))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2})))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3})))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4})))
        disp('no channel files.')
        return
    end

    outputfile= sprintf('%s/%s_round%03i_summedNorm.tif',dst_folder_name,fileroot_name,roundnum);
    if exist(outputfile,'file')
        fprintf('%s already exists, skipping\n',outputfile);
        return
    end

    % Normalize the data
    basename = sprintf('%s_round%03d',fileroot_name,roundnum);
    [chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname,image_height,image_width] = ...
        quantilenorm_small_mem(params.tempDir,basename, ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1})), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2})), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3})), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4})));

    summed_file = sprintf('%s_round%03d_5_summed.bin',fileroot_name,roundnum);
    sumfiles(params.tempDir,{ chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname },summed_file);

    summed_norm = load_binary_image(params.tempDir,summed_file,image_height,image_width);

    save3DTif(summed_norm,outputfile);

end

function image = load_binary_image(outputdir,image_fname,image_height,image_width)
    fid = fopen(fullfile(outputdir,image_fname),'r');
    count = 1;
    while ~feof(fid)
        sub_image = fread(fid,[image_height,image_width],'double');
        if ~isempty(sub_image)
            image(:,:,count) = sub_image;
            count = count + 1;
        end
    end
    fclose(fid);
end

function ret = selectCore(num_core_sem)
    count = 1;
    while true
        ret = semaphore(['/c' num2str(num_core_sem)],'trywait');
        if ret == 0
            fprintf('selectCore[count=%d]\n',count);
            break
        end
        count = count + 1;
        pause(2);
    end
end

function unselectCore(num_core_sem)
    ret = semaphore(['/c' num2str(num_core_sem)],'post');
    if ret == -1
        fprintf('unselect [/c%d] failed.\n',num_core_sem);
    end
end

