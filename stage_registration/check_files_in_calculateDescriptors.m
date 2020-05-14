function [ret,messages] = check_files_in_calculateDescriptors()

    loadParameters;

    if params.DO_DOWNSAMPLE
        filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
    else
        filename_root = sprintf('%s_',params.FILE_BASENAME);
    end

    ret = true;
    messages = {};
    for r_i = 1:params.NUM_ROUNDS
        for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
            descriptor_output_dir = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root,r_i,register_channel{1}));
            file_list = dir([descriptor_output_dir,'*.mat']);

            regChan = register_channel{1};
            img_filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
                filename_root,r_i,regChan,params.IMAGE_EXT));
            if ~exist(img_filename,'file')
                messages{end+1} = sprintf('[ERROR] not created: %s',img_filename);
                continue
            end
            img = load3DImage_uint16(img_filename);

            ymin = 1;
            ymax = size(img,1);
            xmin = 1;
            xmax = size(img,2);

            filename = fullfile(descriptor_output_dir, ...
                [num2str(ymin) '-' num2str(ymax) '_' num2str(xmin) '-' num2str(xmax) '.mat']);
            if ~exist(filename,'file')
                ret = false;
                messages{end+1} = sprintf('[ERROR] not created: %s',filename);
            end
        end
    end
end

