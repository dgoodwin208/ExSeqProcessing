function dim = image_dimensions(filename)

    if endsWith(filename,'.tif')
        imginfo = imfinfo(filename);
        w = imginfo.Width;
        h = imginfo.Height;
        d = length(imginfo);
        dim = [w,h,d];
    elseif endsWith(filename,'.h5')
        imginfo = h5info(filename);
        dim = imginfo.Datasets.Dataspace.Size;
    end
end
