function convertTif2Hdf5(tif_filename)

    img = read_file(tif_filename);

    hdf5_filename = replace(tif_filename,'.tif','.h5');

    if exist(hdf5_filename)
        delete(hdf5_filename);
    end

    tic;
    h5create(hdf5_filename,'/image',size(img),'DataType','uint16');
    h5write(hdf5_filename,'/image',img);
    disp('write as hdf5. '); toc;

end

