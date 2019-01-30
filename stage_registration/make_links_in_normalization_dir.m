function ret = make_links_in_normalization_dir()

    ret = true;

    loadParameters;

    src_dir = relpath(params.normalizedImagesDir,params.colorCorrectionImagesDir);
    cd(params.normalizedImagesDir);

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
            for c_i = 1:length(params.SHIFT_CHAN_STRS)
                src_filename = fullfile(src_dir,sprintf('%s_round%03i_%s.%s',FILEROOT_NAME,r_i,params.SHIFT_CHAN_STRS{c_i},params.IMAGE_EXT));
                if ~exist(src_filename,'file')
                    fprintf('[ERROR] not exist source file: %s\n',src_filename);
                    ret = false;
                    continue
                end
                dst_filename = sprintf('./%s_round%03i_%s.%s',FILEROOT_NAME,r_i,params.SHIFT_CHAN_STRS{c_i},params.IMAGE_EXT);
                if ~exist(dst_filename,'file')
                    fprintf('ln -s %s %s\n',src_filename,dst_filename);
                    system(sprintf('ln -s %s %s',src_filename,dst_filename));
                end
            end
        end
    end

    cd('..');
end

