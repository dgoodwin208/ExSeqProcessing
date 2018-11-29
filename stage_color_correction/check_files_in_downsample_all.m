function [ret,messages] = check_files_in_downsample_all()

    loadParameters;
    orig_chans = {'ch00','ch01','ch02','ch03'};

    ret = true;
    messages = {};
    for rnd_indx = 1:params.NUM_ROUNDS
        for c = 1:params.NUM_CHANNELS

            filename_downsampled = fullfile(params.deconvolutionImagesDir,...
                sprintf('%s-downsample_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,orig_chans{c},params.IMAGE_EXT));

            if ~exist(filename_downsampled,'file')
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',filename_downsampled);
            end

            if isequal(params.IMAGE_EXT,'h5')
                filename_full_hdf5 = fullfile(params.deconvolutionImagesDir,...
                    sprintf('%s_round%.03i_%s.h5',params.FILE_BASENAME,rnd_indx,orig_chans{c}));
                if ~exist(filename_full_hdf5,'file')
                    ret = false;
                    messages{end+1} = sprintf('[ERROR] not created: %s',filename_full_hdf5);
                end
            end
        end
    end
end

