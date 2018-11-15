function ret = postcheck_color_correction(total_round_num)

    loadParameters;

    if params.DO_DOWNSAMPLE
        FILEROOT_NAME = sprintf('%s-downsample',params.FILE_BASENAME);
    else
        FILEROOT_NAME = sprintf('%s',params.FILE_BASENAME);
    end

    postcheck = true;
    for r_i = 1:total_round_num
        for c_i = 2:length(params.CHAN_STRS)
            filename = fullfile(params.colorCorrectionImagesDir,sprintf('%s_round%.03i_%s.%s',FILEROOT_NAME,r_i,params.CHAN_STRS{c_i},params.IMAGE_EXT));
            if ~exist(filename,'file')
                postcheck = false;
                fprintf('[ERROR] not created: %s\n',filename);
            end
        end

        offset_filename = fullfile(params.colorCorrectionImagesDir,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,r_i));
        if ~exist(offset_filename,'file')
            postcheck = false;
            fprintf('[ERROR] not created: %s\n',offset_filename);
        end
    end

    if postcheck
        fprintf('[DONE]\n');
    end

    ret = postcheck;
end

