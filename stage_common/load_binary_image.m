function image = load_binary_image(loaddir,image_fname,image_height,image_width)
    fid = fopen(fullfile(loaddir,image_fname),'r');
    count = 1;
    while ~feof(fid)
        sub_image = fread(fid,[image_height,image_width],'float');
        if ~isempty(sub_image)
            image(:,:,count) = double(sub_image);
            count = count + 1;
        end
    end
    fclose(fid);
end
