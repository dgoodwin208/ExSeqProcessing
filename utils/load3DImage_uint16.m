function vol = load3DImage_uint16(path)

if endsWith(path,'.tif')
    vol = double(read_file(path));
elseif endsWith(path,'.h5')
    vol = double(h5read(path,'/image'));
end

end
