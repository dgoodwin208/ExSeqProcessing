function is_same = compare_tif_with_hdf5_files(tifDir, hdf5Dir)

    tifFiles  = dir(fullfile(tifDir,'*.tif'));
    hdf5Files = dir(fullfile(hdf5Dir,'*.h5'));

    if length(tifFiles) == 0
        disp('no tif file.');
        is_same = false;
        return;
    end

    if length(tifFiles) ~= length(hdf5Files)
        disp('NG - # of files is different.');
        tifFiles
        hdf5Files
        is_same = false;
        return;
    end

    is_same = true;
    for i=1:length(tifFiles)
        tifData  = load3DImage_uint16(fullfile(tifFiles(i).folder,tifFiles(i).name));
        hdf5Data = load3DImage_uint16(fullfile(hdf5Files(i).folder,hdf5Files(i).name));

        if isequal(tifData,hdf5Data)
            disp(['OK - ',tifFiles(i).name])
        else
            disp(['NG - ',tifFiles(i).name])
            is_same = false;
        end
    end

    if is_same
        disp('Total: OK')
    else
        disp('Total: NG')
    end

end

