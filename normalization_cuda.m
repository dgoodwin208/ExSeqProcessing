% normalization

function success_code = normalization_cuda(src_folder_name,dst_folder_name,fileroot_name,channels,total_round_num)

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

    args = {src_folder_name,dst_folder_name,fileroot_name,channels};
    success_code = batch_process('normalization', @normalizeImage, total_round_num, args)

%    quantilenorm_final(length(num_cores));

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

