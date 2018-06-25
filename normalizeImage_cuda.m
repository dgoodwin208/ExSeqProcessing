function normalizeImage_cuda(src_folder_name,dst_folder_name,fileroot_name,channels,roundnum)

    loadParameters;

    image_ext = 'h5';

    if (exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1}, image_ext))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2}, image_ext))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3}, image_ext))) || ...
        exist(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4}, image_ext))))
    else
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1}, image_ext)))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2}, image_ext)))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3}, image_ext)))
        disp(fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4}, image_ext)))
        disp('no channel files.')
        return
    end

    outputfile= sprintf('%s/%s_round%03i_summedNorm.%s',dst_folder_name,fileroot_name,roundnum, image_ext);
    if exist(outputfile,'file')
        fprintf('%s already exists, skipping\n',outputfile);
        return
    end

    % Normalize the data
    basename = sprintf('%s_round%03d',fileroot_name,roundnum);
    ret = ...
        quantilenorm_cuda(params.tempDir,basename, { ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{1}, image_ext)), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{2}, image_ext)), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{3}, image_ext)), ...
        fullfile(src_folder_name,sprintf('%s_round%.03i_%s.%s',fileroot_name,roundnum,channels{4}, image_ext)) });

    disp(ret)

    chan1_norm_fname = ret{1};
    chan2_norm_fname = ret{2};
    chan3_norm_fname = ret{3};
    chan4_norm_fname = ret{4};
    image_size = ret{5};
    image_height = image_size(1);
    image_width  = image_size(2);
%    num_slices   = image_size(3);

    summed_file = sprintf('%s_round%03d_5_summed.bin',fileroot_name,roundnum);
    sumbinfiles(params.tempDir,{ chan1_norm_fname,chan2_norm_fname,chan3_norm_fname,chan4_norm_fname },summed_file);

    summed_norm = load_binary_image(params.tempDir,summed_file,image_height,image_width);

    if strcmp(image_ext,'tif')
        save3DTif_uint16(summed_norm,outputfile);
    elseif strcmp(image_ext,'h5')
        h5create(outputfile,'/image',image_size,'DataType','uint16');
        h5write(outputfile,'/image',uint16(summed_norm));
    end

end

function image = load_binary_image(outputdir,image_fname,image_height,image_width)
    fid = fopen(fullfile(outputdir,image_fname),'r');
    count = 1;
    while ~feof(fid)
        sub_image = fread(fid,[image_height,image_width],'double');
        if ~isempty(sub_image)
            image(:,:,count) = sub_image;
            count = count + 1;
        end
    end
    fclose(fid);
end

