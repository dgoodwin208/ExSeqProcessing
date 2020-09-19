function normalizeImage_cuda(roundnum,do_downsample)
    %TODO: The CUDA Version of normalization still needs to be improved. See normalizeImage.m 
    % for reference. -DG 2020 09 09
    loadParameters;

    src_folder_name = params.colorCorrectionImagesDir;
    dst_folder_name = params.normalizedImagesDir;
    channels = params.SHIFT_CHAN_STRS;

    if do_downsample
        fileroot_name = sprintf('%s-downsample',params.FILE_BASENAME);
    else
        fileroot_name = sprintf('%s',params.FILE_BASENAME);
    end

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

    use_tmp_files = true;

    % Normalize the data
    basename = sprintf('%s_round%03d',fileroot_name,roundnum);
    ret = ...
        quantilenorm_cuda(params.tempDir,basename, { ...
            fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1},params.IMAGE_EXT)), ...
            fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2},params.IMAGE_EXT)), ...
            fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3},params.IMAGE_EXT)), ...
            fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4},params.IMAGE_EXT)) }, use_tmp_files);

    chan1_norm_fname = ret{1};
    chan2_norm_fname = ret{2};
    chan3_norm_fname = ret{3};
    chan4_norm_fname = ret{4};
    image_size = ret{5};
    image_height = image_size(1);
    image_width  = image_size(2);
%    num_slices   = image_size(3);

    summed_file = sprintf('%s_round%03d_5_summed.bin',fileroot_name,roundnum);
    sumbinfiles(params.tempDir,{ chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname },summed_file);

    summed_norm_image = load_binary_image(params.tempDir,summed_file,image_height,image_width);

    save3DImage_uint16(summed_norm_image,outputfile);
    clear summed_norm_image

    tic;
    delete(fullfile(params.tempDir,chan1_norm_fname), ...
           fullfile(params.tempDir,chan2_norm_fname), ...
           fullfile(params.tempDir,chan3_norm_fname), ...
           fullfile(params.tempDir,chan4_norm_fname), ...
           fullfile(params.tempDir,summed_file));
    disp('delete the rest temp files'); toc;

end

