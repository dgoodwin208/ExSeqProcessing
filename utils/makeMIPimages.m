% makeMIPimages
%     make MIP movies from 3D images
%
% ex)
% inputFile=dir('4_registration/*_round*_summedNorm_registered.tif')
% makeMIPimages(inputFile, 'registration-mip.avi')

function makeMIPimages(inputFiles, outputFile)

    num_input = length(inputFiles);

    a_z = {};
    a_c = {};
    a_r = {};
    for i = 1:num_input
        d = load3DImage_uint16(fullfile(inputFiles(i).folder,inputFiles(i).name));

        if i == 1
            size_r = size(d,1);
            size_c = size(d,2);
            size_z = size(d,3);
        end

        % MIPs along the z-axis
        a_z{i} = imadjust(max(d,[],3)/65535);
        % MIPs along the y-axis
        a_c{i} = imadjust(reshape(max(d,[],2)/65535,[size_r,size_z]));
        % MIPs along the x-axis
        a_r{i} = imadjust(reshape(max(d,[],1)/65535,[size_c,size_z]));
    end

    if length(d) == 0
        disp('no file.');
        return;
    end


    m_z = reshape(cell2mat(a_z),[size_r,size_c,1,num_input]);
    m_c = reshape(cell2mat(a_c),[size_r,size_z,1,num_input]);
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


    % diff images (imfuse)
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


    % diff images (imabsdiff)
    absdiffOutputFile = [outputFile(1:end-4),'-absdiff.avi'];
    v = VideoWriter(absdiffOutputFile,'Uncompressed AVI');
    v.FrameRate = 2;
    open(v);

    gap = 20;
    for i = 2:num_input
        diff_m_z = imabsdiff(m_z(:,:,1,1),m_z(:,:,1,i));
        diff_m_c = imabsdiff(m_c(:,:,1,1),m_c(:,:,1,i));
        diff_m_r = imabsdiff(m_r(:,:,1,1),m_r(:,:,1,i));
        sub_image = ones(size_r+gap+size_z,size_c+gap+size_z);
        sub_image(1:size_r,1:size_c) = diff_m_z;
        sub_image(1:size_r,size_c+gap+1:size_c+gap+size_z) = diff_m_c;
        sub_image(size_r+gap+1:size_r+gap+size_z,1:size_c) = permute(diff_m_r,[2,1]);
        writeVideo(v,sub_image);
    end
    close(v);
end

