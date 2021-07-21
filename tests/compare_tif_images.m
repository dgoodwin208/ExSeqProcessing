function is_same = compare_tif_images(srcDir, dstDir, avgErrTol, postfix)

    srcFiles = dir(fullfile(srcDir,['*', postfix, '.tif']));
    dstFiles = dir(fullfile(dstDir,['*', postfix, '.tif']));

    if length(srcFiles) == 0
        disp('no tif file.');
        is_same = false;
        return;
    end

    if length(srcFiles) ~= length(dstFiles)
        disp('NG - # of files is different.');
        srcFiles
        dstFiles
        is_same = false;
        return;
    end

    format shortE
    hst_edges = [0 1e-5 1e-4 1e-3 1e-2 1e-1 1];
    avgErrList = [];
    is_same = true;
    for i=1:length(srcFiles)
        disp(fullfile(srcFiles(i).folder,srcFiles(i).name))
        srcImg = single(load3DImage_uint16(fullfile(srcFiles(i).folder,srcFiles(i).name))) / 65535;
        dstImg = single(load3DImage_uint16(fullfile(dstFiles(i).folder,dstFiles(i).name))) / 65535;

        %oneMat = ones(size(srcImg));
        absMat = abs(srcImg - dstImg);
        avgErr = mean(absMat,'all');
        disp(['##### Avg of Err: ',num2str(avgErr)])
        if avgErr > avgErrTol
            is_same = false;
        end
        avgErrList = horzcat(avgErrList, avgErr);

        h = histogram(absMat(:),hst_edges);
        num_elms = size(h.Data(:));
        fprintf('# of elms = %d\n',num_elms(1))
        disp(h.Values)
        disp(h.BinEdges)
    end

    disp('##### Avg of Err list:')
    disp(avgErrList)

    if is_same
        disp('Total: OK')
    else
        disp('Total: NG')
    end

end

