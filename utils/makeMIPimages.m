function makeMIPimages(inputFiles, outputFile)

    num_input = length(inputFiles);

    d = {};
    for i = 1:num_input
        d{i} = load3DTif_uint16(fullfile(inputFiles(i).folder,inputFiles(i).name));
    end

    if length(d) == 0
        disp('no file.');
        return;
    end

    size_r = size(d{1},1);
    size_c = size(d{1},2);
    size_z = size(d{1},3);


    % MIPs along the z-axis
    a_z = {};
    for i = 1:num_input
        a_z{i} = imadjust(max(d{i},[],3)/65535);
    end

    m_z = reshape(cell2mat(a_z),[size_r,size_c,1,num_input]);


    % MIPs along the y-axis
    a_c = {};
    for i = 1:num_input
        a_c{i} = imadjust(reshape(max(d{i},[],2)/65535,[size_r,size_z]));
    end

    m_c = reshape(cell2mat(a_c),[size_r,size_z,1,num_input]);


    % MIPs along the x-axis
    a_r = {};
    for i = 1:num_input
        a_r{i} = imadjust(reshape(max(d{i},[],1)/65535,[size_c,size_z]));
    end

    m_r = reshape(cell2mat(a_r),[size_c,size_z,1,num_input]);


    v = VideoWriter(outputFile,'Uncompressed AVI');
    v.FrameRate = 2;
    open(v);

    gap = 20;
    for i = 1:num_input
        sub_image = ones(size_r+gap+size_z,size_c+gap+size_z);
        sub_image(1:size_r,1:size_c) = m_z(:,:,1,i);
        sub_image(1:size_r,size_c+gap+1:size_c+gap+size_z) = m_c(:,:,1,i);
        sub_image(size_r+gap+1:size_r+gap+size_z,1:size_c) = m_r(:,:,1,i)';
        writeVideo(v,sub_image);
    end
    close(v);


    % diff images
    diffOutputFile = [outputFile(1:end-4),'-diff.avi'];
    v = VideoWriter(diffOutputFile,'Uncompressed AVI');
    v.FrameRate = 2;
    open(v);

    gap = 20;
    for i = 1:num_input
        diff_m_z = imfuse(m_z(:,:,1,1),m_z(:,:,1,i));
        diff_m_c = imfuse(m_c(:,:,1,1),m_c(:,:,1,i));
        diff_m_r = imfuse(m_r(:,:,1,1),m_r(:,:,1,i));
        sub_image = ones(size_r+gap+size_z,size_c+gap+size_z,3);
        sub_image(1:size_r,1:size_c,:) = diff_m_z;
        sub_image(1:size_r,size_c+gap+1:size_c+gap+size_z,:) = diff_m_c;
        sub_image(size_r+gap+1:size_r+gap+size_z,1:size_c,:) = permute(diff_m_r,[2,1,3]);
        writeVideo(v,sub_image/255);
    end
    close(v);
end

