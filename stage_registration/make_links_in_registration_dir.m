function ret = make_links_in_registration()

    ret = true;

    loadParameters;

    src_dir = relpath(params.registeredImagesDir,params.normalizedImagesDir);
    cd(params.registeredImagesDir);

    for c_i = 1:length(regparams.CHANNELS)
        normalized_filename = fullfile(src_dir,sprintf('%s_round%03i_%s.%s',params.FILE_BASENAME,params.REFERENCE_ROUND_WARP,regparams.CHANNELS{c_i},params.IMAGE_EXT));
        if ~exist(normalized_filename,'file')
            fprintf('[ERROR] not exist source file: %s\n',normalized_filename);
            ret = false;
            continue
        end
        reg_affine_filename = sprintf('./%s_round%03i_%s_affine.%s',params.FILE_BASENAME,params.REFERENCE_ROUND_WARP,regparams.CHANNELS{c_i},params.IMAGE_EXT);
        if ~exist(reg_affine_filename,'file')
            fprintf('ln -s %s %s\n',normalized_filename,reg_affine_filename);
            system(sprintf('ln -s %s %s',normalized_filename,reg_affine_filename));
        end

        if strcmp(regparams.REGISTRATION_TYPE,'affine')
            continue
        end

        reg_tps_filename = sprintf('./%s_round%03i_%s_registered.%s',params.FILE_BASENAME,params.REFERENCE_ROUND_WARP,regparams.CHANNELS{c_i},params.IMAGE_EXT);
        if ~exist(reg_tps_filename,'file')
            fprintf('ln -s %s %s\n',normalized_filename,reg_tps_filename);
            system(sprintf('ln -s %s %s',normalized_filename,reg_tps_filename));
        end
    end

    cd('..');
end

