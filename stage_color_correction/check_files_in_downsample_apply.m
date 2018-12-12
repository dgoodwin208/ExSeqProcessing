function [ret,messages] = check_files_in_downsample_apply()

    loadParameters;

    ret = true;
    messages = {};
    for rnd_indx = 1:params.NUM_ROUNDS
        for c = 1:params.NUM_CHANNELS
            chan_outname = fullfile(params.colorCorrectionImagesDir,...
            sprintf('%s_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,params.SHIFT_CHAN_STRS{c},params.IMAGE_EXT));
            if ~exist(chan_outname)
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',chan_outname);
            end
        end
    end
end

