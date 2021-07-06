loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    fprintf('[DONE]\n');
    return;
end

for rnd_indx = 1:params.NUM_ROUNDS
    precheck = true;
    for c = 1:params.NUM_CHANNELS

        filename_full = fullfile(params.deconvolutionImagesDir,...
            sprintf('%s_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c},params.INPUT_IMAGE_EXT));

        if ~exist(filename_full,'file')
            precheck = false;
            fprintf('[ERROR] missing file: %s\n', filename_full);
        end
    end

    if ~precheck
        exit
    end
end

delete(gcp('nocreate'))
conditions = conditions_for_concurrency();
max_pool_size = concurrency_size_in_downsample_all(conditions);

parpool(max_pool_size);

if ~isequal(params.INPUT_IMAGE_EXT,params.IMAGE_EXT)
    fprintf('Converting input images: %s to %s\n',params.INPUT_IMAGE_EXT,params.IMAGE_EXT)

    parfor rnd_indx = 1:params.NUM_ROUNDS
        num_channels = params.NUM_CHANNELS;
        chan_strs = params.CHAN_STRS;
        % NOTE: In the ExSeq experiments, we would image all the sequencing
        % channels AND a morphology channel. So that is what is referenced
        % here, and additional channel number, ie, ch_04, would be added. 
        if isfield(params, 'MORPHOLOGY_ROUND') && (rnd_indx == params.MORPHOLOGY_ROUND)
            num_channels = num_channels + 1;
            chan_strs{num_channels} = params.MORPHOLOGY_CHAN_STR;
        end

        for c = 1:num_channels

            input_filename_full = fullfile(params.deconvolutionImagesDir,...
                sprintf('%s_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,chan_strs{c},params.INPUT_IMAGE_EXT));
            filename_full = fullfile(params.deconvolutionImagesDir,...
                sprintf('%s_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,chan_strs{c},params.IMAGE_EXT));

            img = load3DImage_uint16(input_filename_full);
            if ~exist(filename_full,'file')
                fprintf('Saving %s\n',filename_full);
                save3DImage_uint16(img,filename_full);
            end
        end
    end
end

parfor rnd_indx = 1:params.NUM_ROUNDS
    num_channels = params.NUM_CHANNELS;
    chan_strs = params.CHAN_STRS;
    if isfield(params, 'MORPHOLOGY_ROUND') && (rnd_indx == params.MORPHOLOGY_ROUND)
        num_channels = num_channels + 1;
        chan_strs{num_channels} = params.MORPHOLOGY_CHAN_STR;
    end

    for c = 1:num_channels

        filename_full = fullfile(params.deconvolutionImagesDir,...
            sprintf('%s_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,chan_strs{c},params.IMAGE_EXT));
        filename_downsampled = fullfile(params.deconvolutionImagesDir,...
            sprintf('%s-downsample_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,chan_strs{c},params.IMAGE_EXT));

        if exist(filename_downsampled,'file')
            fprintf('Skipping file %s that already exists\n',filename_downsampled);
            continue;
        end

        img = load3DImage_uint16(filename_full);

        %Doing 'linear' downsample because the default of cubic was creating
        %values as low as -76
        img_downsample = imresize3(img,1/params.DOWNSAMPLE_RATE,'linear');

        fprintf('Saving %s \n',filename_downsampled);
        save3DImage_uint16(img_downsample,filename_downsampled);

    end
end

