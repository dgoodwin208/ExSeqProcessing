function ret = postcheck_registerWithCorrespondences(total_round_num)

    loadParameters;

    postcheck = true;
    for do_downsample = [true, false]
        if do_downsample
            filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
        else
            filename_root = sprintf('%s',params.FILE_BASENAME);
        end

        for r_i = 1:total_round_num
            if r_i == params.REFERENCE_ROUND_WARP
                continue
            end

            output_affine_filename = fullfile(regparams.OUTPUTDIR,sprintf('%s_round%03d_%s_affine.%s',filename_root,r_i,regparams.CHANNELS{end},params.IMAGE_EXT));
            if ~exist(output_affine_filename,'file')
                postcheck = false;
                fprintf('[ERROR] not created: %s\n',output_affine_filename);
            end

            output_keys_filename = fullfile(regparams.OUTPUTDIR,sprintf('globalkeys_%s-downsample_round%03d.mat',params.FILE_BASENAME,r_i));
            if ~exist(output_keys_filename,'file')
                postcheck = false;
                fprintf('[ERROR] not created: %s\n',output_keys_filename);
            end
        end
    end

    if strcmp(regparams.REGISTRATION_TYPE,'affine')
        if postcheck
            fprintf('[DONE]\n');
        end
        ret = postcheck;

        fprintf('Ending the registration after the affine\n');
        return;
    end


    postcheck = true;
    for do_downsample = [true, false]
        if do_downsample
            filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
        else
            filename_root = sprintf('%s',params.FILE_BASENAME);
        end

        for r_i = 1:total_round_num
            if r_i == params.REFERENCE_ROUND_WARP
                continue
            end

            % TODO: unify hdf5
            output_TPS_filename = fullfile(regparams.OUTPUTDIR,sprintf('TPSMap_%s_round%03d.h5',filename_root,r_i));
            if ~exist(output_TPS_filename,'file')
                output_TPS_filename = fullfile(regparams.OUTPUTDIR,sprintf('TPSMap_%s_round%03d.mat',filename_root,r_i));
                if ~exist(output_TPS_filename,'file')
                    postcheck = false;
                    fprintf('[ERROR] not created: %s\n',output_TPS_filename);
                end
            end

            for c = 1:length(regparams.CHANNELS)
                data_channel = regparams.CHANNELS{c};
                outputfile = fullfile(regparams.OUTPUTDIR,sprintf('%s_round%03d_%s_%s.%s',filename_root,r_i,data_channel,regparams.REGISTRATION_TYPE,params.IMAGE_EXT));
                if ~exist(outputfile,'file')
                    postcheck = false;
                    fprintf('[ERROR] not created: %s\n',outputfile);
                end
            end
        end
    end

    if postcheck
        fprintf('[DONE]\n');
    end

    ret = postcheck;
end

