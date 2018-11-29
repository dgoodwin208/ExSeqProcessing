function [ret,messages] = check_files_in_registerWithCorrespondences()

    loadParameters;

    ret = true;
    messages = {};
    for do_downsample = [true, false]
        if do_downsample
            filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
        else
            filename_root = sprintf('%s',params.FILE_BASENAME);
        end

        for r_i = 1:params.NUM_ROUNDS
            if r_i == params.REFERENCE_ROUND_WARP
                continue
            end

            output_affine_filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.%s',filename_root,r_i,regparams.CHANNELS{end},params.IMAGE_EXT));
            if ~exist(output_affine_filename,'file')
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',output_affine_filename);
            end

            output_keys_filename = fullfile(params.registeredImagesDir,sprintf('globalkeys_%s-downsample_round%03d.mat',params.FILE_BASENAME,r_i));
            if ~exist(output_keys_filename,'file')
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',output_keys_filename);
            end
        end
    end

    if ~ret
        return;
    else
        if strcmp(regparams.REGISTRATION_TYPE,'affine')
            messages{end+1} = sprintf('[DONE]');
            return;
        end
    end


    ret = true;
    for do_downsample = [true, false]
        if do_downsample
            filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
        else
            filename_root = sprintf('%s',params.FILE_BASENAME);
        end

        for r_i = 1:params.NUM_ROUNDS
            if r_i == params.REFERENCE_ROUND_WARP
                continue
            end

            % TODO: unify hdf5
            output_TPS_filename = fullfile(params.registeredImagesDir,sprintf('TPSMap_%s_round%03d.h5',filename_root,r_i));
            if ~exist(output_TPS_filename,'file')
                output_TPS_filename = fullfile(params.registeredImagesDir,sprintf('TPSMap_%s_round%03d.mat',filename_root,r_i));
                if ~exist(output_TPS_filename,'file')
                    ret = false;
                    messages{end+1} = sprintf('[ERROR] not created: %s',output_TPS_filename);
                end
            end

            for c = 1:length(regparams.CHANNELS)
                data_channel = regparams.CHANNELS{c};
                outputfile = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_%s.%s',filename_root,r_i,data_channel,regparams.REGISTRATION_TYPE,params.IMAGE_EXT));
                if ~exist(outputfile,'file')
                    ret = false;
                    messages{end+1} = sprintf('[ERROR] not created: %s',outputfile);
                end
            end
        end
    end
end

