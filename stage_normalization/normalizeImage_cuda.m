function normalizeImage_cuda(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum)

    loadParameters;

    %Check for any missing image file
    chan_filenames = {};
    for c_idx = 1:length(channels)
        filename = fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',...
            fileroot_name,roundnum,channels{c_idx},params.IMAGE_EXT));
        if ~exist(filename,'file')
            fprintf('Missing channel file: %s\n',filename);
        end
        chan_filenames{end+1} = filename;
    end

    outputfile= sprintf('%s/%s_round%03i_summedNorm.%s',dst_folder_name,fileroot_name,roundnum,params.IMAGE_EXT);
    if exist(outputfile,'file')
        fprintf('%s already exists, skipping\n',outputfile);
        return
    end

    use_tmp_files = true;

    % Normalize the data
    basename = sprintf('%s_round%03d',fileroot_name,roundnum);
    ret = quantilenorm_cuda(params.tempDir,basename,chan_filenames,use_tmp_files);

    chan_norm_filenames = {};
    for c_idx = 1:length(channels)
        chan_norm_filenames{end+1} = ret{c_idx};
    end
    image_size = ret{length(channels)+1};
    image_height = image_size(1);
    image_width  = image_size(2);
%    num_slices   = image_size(3);

    summed_file = sprintf('%s_round%03d_5_summed.bin',fileroot_name,roundnum);
    sumbinfiles(params.tempDir,chan_norm_filenames,summed_file);

    summed_norm_image = load_binary_image(params.tempDir,summed_file,image_height,image_width);

    save3DImage_uint16(summed_norm_image,outputfile);
    clear summed_norm_image

    tic;
    for c_idx = 1:length(channels)
        delete(fullfile(params.tempDir,chan_norm_filenames{c_idx}));
    end
    delete(fullfile(params.tempDir,summed_file));
    disp('delete the rest temp files'); toc;

end

