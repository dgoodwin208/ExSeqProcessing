function normalizeImage(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum)

    if (exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3}))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4}))))
    else
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1})))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2})))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3})))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4})))
        disp('no channel files.')
        return
    end

    outputfile= sprintf('%s/%s_round%03i_summedNorm.tif',dst_folder_name,fileroot_name,roundnum);
    if exist(outputfile,'file')
        fprintf('%s already exists, skipping\n',outputfile);
        return
    end
    chan1 = load3DTif_uint16(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{1})));
    chan2 = load3DTif_uint16(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{2})));
    chan3 = load3DTif_uint16(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{3})));
    chan4 = load3DTif_uint16(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.tif',fileroot_name,roundnum,channels{4})));
    size_chan1 = size(chan1);
    size_chan2 = size(chan2);
    size_chan3 = size(chan3);
    size_chan4 = size(chan4);

    data_cols(:,1) = reshape(chan1,[],1);
    data_cols(:,2) = reshape(chan2,[],1);
    data_cols(:,3) = reshape(chan3,[],1);
    data_cols(:,4) = reshape(chan4,[],1);
    clearvars chan1 chan2 chan3 chan4;

    % Normalize the data
    data_cols_norm = quantilenorm(data_cols);
    clearvars data_cols;

    % reshape the normed results back into 3d images
    chan1_norm = reshape(data_cols_norm(:,1),size_chan1);
    chan2_norm = reshape(data_cols_norm(:,2),size_chan2);
    chan3_norm = reshape(data_cols_norm(:,3),size_chan3);
    chan4_norm = reshape(data_cols_norm(:,4),size_chan4);
    clearvars data_cols_norm;


    summed_norm = chan1_norm+chan2_norm+chan3_norm+chan4_norm;
    clearvars chan1_norm chan2_norm chan3_norm chan4_norm;

    save3DTif_uint16(summed_norm,outputfile);

end

