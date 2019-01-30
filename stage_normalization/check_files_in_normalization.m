function [ret,messages] = check_files_in_normalization()

    loadParameters;

    ret = true;
    messages = {};
    for do_downsample = [true,false]
        if do_downsample && ~params.DO_DOWNSAMPLE
            continue
        end

        if do_downsample
            FILEROOT_NAME = sprintf('%s-downsample',params.FILE_BASENAME);
        else
            FILEROOT_NAME = sprintf('%s',params.FILE_BASENAME);
        end

        for r_i = 1:params.NUM_ROUNDS
            filename = fullfile(params.normalizedImagesDir,sprintf('%s_round%03i_summedNorm.%s',FILEROOT_NAME,r_i,params.IMAGE_EXT));
            if ~exist(filename,'file')
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',filename);
            end
        end
    end
end

