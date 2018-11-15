function ret = postcheck_normalization(total_round_num,do_downsample)

    loadParameters;

    if do_downsample
        FILEROOT_NAME = sprintf('%s-downsample',params.FILE_BASENAME);
    else
        FILEROOT_NAME = sprintf('%s',params.FILE_BASENAME);
    end

    postcheck = true;
    for r_i = 1:total_round_num
        filename = sprintf('%s/%s_round%03i_summedNorm.%s',params.normalizedImagesDir,FILEROOT_NAME,r_i,params.IMAGE_EXT);
        if ~exist(filename,'file')
            postcheck = false;
            fprintf('[ERROR] not created: %s\n',filename);
        end
    end

    if postcheck
        fprintf('[DONE]\n');
    end

    ret = postcheck;
end

