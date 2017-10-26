% normalization

function normalization_cuda(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num)

    loadParameters;

    if length(channels) ~= 4
        disp('# of channels is not 4.')
        return
    end

    cluster = parcluster('local_200workers');
    %parpool(cluster);

%    num_sem_gpus = [1,1];
    num_sem_gpus = ones(1, gpuDeviceCount());
%    num_sem_cores = [20, 10, 15, 1];
    num_sem_cores = [params.NORM_JOB_SIZE, 0, 20, 0];
    quantilenorm_cuda_init(num_sem_gpus,num_sem_cores);

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
               'CaptureDiary',true);
%            normalizeImage(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum); % serial run
            roundnum = roundnum+1;
        else
            for job_id = find(running_jobs==1)
                job = jobs{job_id};
                is_finished = 0;
                if strcmp(job.State,'finished')
                    disp(['batch (',num2str(job_id),') has ',job.State,'.'])
                    diary(job,['./matlab-normalization-',num2str(job_id),'.log']);
                    running_jobs(job_id) = 0;
                    delete(job)
                    is_finished = 1;
                elseif strcmp(job.State,'failed')
                    disp(['batch (',num2str(job_id),') has ',job.State,', resubmit it.'])
                    diary(job,['./matlab-normalization-',num2str(job_id),'-failed.log']);
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

    disp('===== all batch jobs finished')
    toc;

%    quantilenorm_final(length(num_cores));

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
    ret = ...
        quantilenorm_cuda(params.tempDir,basename, { ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1})), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2})), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3})), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4})) });

    chan1_norm_fname = ret{1};
    chan2_norm_fname = ret{2};
    chan3_norm_fname = ret{3};
    chan4_norm_fname = ret{4};
    image_height = ret{5};
    image_width  = ret{6};

    summed_file = sprintf('%s_round%03d_5_summed.bin',fileroot_name,roundnum);
    sumbinfiles(params.tempDir,{ chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname },summed_file);

    summed_norm = load_binary_image(params.tempDir,summed_file,image_height,image_width);

    save3DTif_uint16(summed_norm,outputfile);

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

