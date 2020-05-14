function [imgsize,dim] = imagesize_in_double(filename)

    imginfo = imfinfo(filename);

    filesize = imginfo.FileSize;
    bitdepth = imginfo.BitDepth;

    w = imginfo.Width;
    h = imginfo.Height;
    d = length(imginfo);
    dim = [w,h,d];

    if (bitdepth ~= 64)
        bytes = bitdepth / 8;
        filesize = filesize * (8 / bytes);
    end

    imgsize = filesize / 1024 / 1024;
end
