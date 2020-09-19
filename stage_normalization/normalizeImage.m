function normalizeImage(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum)

    loadParameters;

    %Check for any missing image file
    for c_idx = 1:length(channels)
        filename = fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',...
            fileroot_name,roundnum,channels{c_idx},params.IMAGE_EXT));
        if ~exist(filename,'file')
            fprintf('Missing channel file: %s\n',filename);
        end
    end

    outputfile= sprintf('%s/%s_round%03i_summedNorm.%s',dst_folder_name,fileroot_name,roundnum,params.IMAGE_EXT);
    if exist(outputfile,'file')
        fprintf('%s already exists, skipping\n',outputfile);
        return
    end
    
    %Load one channel to get the dimensions correct
    chan1 = load3DImage_uint16(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1},params.IMAGE_EXT)));
    numpixels=numel(chan1);    
    size_chan1 = size(chan1);
    
    % Insert the first color channel into the data_cols object for
    % normalizatoin
    data_cols = zeros(numpixels,length(channels));
    data_cols(:,1) = reshape(chan1,[],1);
    % Insert the rest of the color channels
    for c_idx = 2:length(channels)
        chan = load3DImage_uint16(fullfile(src_folder_name,...
            sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,...
            channels{c_idx},params.IMAGE_EXT)));
        data_cols(:,c_idx) = reshape(chan,[],1);
    end

    % Normalize the data
    data_cols_norm = quantilenorm(data_cols);
    clearvars data_cols;

    % reshape the normed results back into 3d images
    summed_norm = zeros(size_chan1);
    for c_idx = 1:length(channels)
        chan_norm = reshape(data_cols_norm(:,c_idx),size_chan1);
        summed_norm = chan_norm + summed_norm;
    end
    
    clearvars data_cols_norm;

    clearvars chan1_norm chan2_norm chan3_norm chan4_norm;

    save3DImage_uint16(summed_norm,outputfile);

end
