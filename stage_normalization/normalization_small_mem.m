% normalization

function normalization_small_mem(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num)

    loadParameters;

    if length(channels) ~= 4
        disp('# of channels is not 4.')
        return
    end

    cluster = parcluster('local_logical_cores');
    %parpool(cluster);

    num_cores = [40 20];
    %num_cores = [10 5];
    %num_cores = [5 5];
    quantilenorm_init(5,num_cores);

    tic;
    disp('===== create batch jobs')

    max_running_jobs = params.NORM_MAX_RUN_JOBS;
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
               'Pool',params.NORM_MAX_POOL_SIZE,'CaptureDiary',true);
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
                       'Pool',params.NORM_MAX_POOL_SIZE,'CaptureDiary',true);
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

    if (exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1},params.IMAGE_EXT))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2},params.IMAGE_EXT))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3},params.IMAGE_EXT))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4},params.IMAGE_EXT))))
    else
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1},params.IMAGE_EXT)))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2},params.IMAGE_EXT)))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3},params.IMAGE_EXT)))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4},params.IMAGE_EXT)))
        disp('no channel files.')
        return
    end

    outputfile= sprintf('%s/%s_round%03i_summedNorm.%s',dst_folder_name,fileroot_name,roundnum,params.IMAGE_EXT);
    if exist(outputfile,'file')
        fprintf('%s already exists, skipping\n',outputfile);
        return
    end

    % Normalize the data
    basename = sprintf('%s_round%03d',fileroot_name,roundnum);
    [chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname,image_height,image_width] = ...
        quantilenorm_small_mem(params.tempDir,basename, ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1},params.IMAGE_EXT)), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2},params.IMAGE_EXT)), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3},params.IMAGE_EXT)), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4},params.IMAGE_EXT)));

    summed_file = sprintf('%s_round%03d_5_summed.bin',fileroot_name,roundnum);
    sumfiles(params.tempDir,{ chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname },summed_file);

    summed_norm = load_binary_image(params.tempDir,summed_file,image_height,image_width);

    save3DImage_uint16(summed_norm,outputfile);

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
