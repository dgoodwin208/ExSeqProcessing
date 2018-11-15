function ret = postcheck_calculateDescriptors(total_round_num)

    loadParameters;

    if params.DO_DOWNSAMPLE
        filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
    else
        filename_root = sprintf('%s_',params.FILE_BASENAME);
    end

    postcheck = true;
    for r_i = 1:total_round_num
        for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
            descriptor_output_dir = fullfile(regparams.OUTPUTDIR,sprintf('%sround%03d_%s/',filename_root,r_i,register_channel{1}));
            file_list = dir([descriptor_output_dir,'*.mat']);

            if length(file_list) ~= regparams.COLS_DESC * regparams.ROWS_DESC
                postcheck = false;

                regChan = register_channel{1};
                img_filename = fullfile(regparams.INPUTDIR,sprintf('%sround%03d_%s.%s',...
                    filename_root,r_i,regChan,params.IMAGE_EXT));
                if ~exist(filename,'file')
                    fprintf('[ERROR] not created: %s\n',img_filename);
                    continue
                end
                img = load3DImage_uint16(img_filename);

                tile_upperleft_y = floor(linspace(1,size(img,1),regparams.ROWS_DESC+1));
                tile_upperleft_x = floor(linspace(1,size(img,2),regparams.COLS_DESC+1));

                for x_idx=1:regparams.COLS_DESC
                    for y_idx=1:regparams.ROWS_DESC
                        % get region, indexing column-wise
                        ymin = tile_upperleft_y(y_idx);
                        ymax = tile_upperleft_y(y_idx+1);
                        xmin = tile_upperleft_x(x_idx);
                        xmax = tile_upperleft_x(x_idx+1);

                        filename = fullfile(descriptor_output_dir, ...
                            [num2str(ymin) '-' num2str(ymax) '_' num2str(xmin) '-' num2str(xmax) '.mat']);
                        if ~exist(filename,'file')
                            fprintf('[ERROR] not created: %s\n',filename);
                        end
                    end
                end
            end
        end
    end

    if postcheck
        fprintf('[DONE]\n');
    end

    ret = postcheck;
end

