function [ret,messages] = check_files_in_color_correction()

    loadParameters;

    if params.DO_DOWNSAMPLE
        FILEROOT_NAME = sprintf('%s-downsample',params.FILE_BASENAME);
    else
        FILEROOT_NAME = sprintf('%s',params.FILE_BASENAME);
    end

    ret = true;
    messages = {};
    for r_i = 1:params.NUM_ROUNDS
        for c_i = 2:length(params.CHAN_STRS)
            filename = fullfile(params.colorCorrectionImagesDir,sprintf('%s_round%.03i_%s.%s',FILEROOT_NAME,r_i,params.CHAN_STRS{c_i},params.IMAGE_EXT));
            if ~exist(filename,'file')
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',filename);
            end
        end

        offset_filename = fullfile(params.colorCorrectionImagesDir,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,r_i));
        if ~exist(offset_filename,'file')
            ret = false;
            messages{end+1} = sprintf('[ERROR] not created: %s',offset_filename);
        end
    end
end

