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
