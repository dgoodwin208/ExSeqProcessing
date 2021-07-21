function is_same = compare_hdf5_images(srcDir, dstDir, ds, avgErrTol)

    srcFiles = dir(fullfile(srcDir,'*.h5'));
    dstFiles = dir(fullfile(dstDir,'*.h5'));

    if length(srcFiles) == 0
        disp('no hdf5 file.');
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
    hst_edges = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10];
    is_same = true;
    for i=1:length(srcFiles)
        disp(fullfile(srcFiles(i).folder,srcFiles(i).name))
        srcImg = single(h5read(fullfile(srcFiles(i).folder,srcFiles(i).name),ds)) / 65535;
        dstImg = single(h5read(fullfile(dstFiles(i).folder,dstFiles(i).name),ds)) / 65535;

        %oneMat = ones(size(srcImg));
        absMat = abs(srcImg - dstImg);
        avgErr = mean(absMat,'all');
        disp(['##### Avg of Err: ',num2str(avgErr)])
        if avgErr > avgErrTol
            is_same = false;
        end

        h = histogram(absMat(:),hst_edges);
        num_elms = size(h.Data(:));
        fprintf('# of elms = %d\n',num_elms(1))
        disp(h.Values)
        disp(h.BinEdges)
%         disp('##### Tol: 1e-6')
%         if isequal((absMat <= 1e-6), oneMat)
%             disp(['OK - ',srcFiles(i).name])
%         else
%             disp(['NG - ',srcFiles(i).name])
%             %is_same = false;
%         end
% %        if isequal(ismembertol(srcImg,dstImg), oneMat)
% %            disp(['OK - ',srcFiles(i).name])
% %        else
% %            disp(['NG - ',srcFiles(i).name])
% %            %is_same = false;
% %        end
%         disp('##### Tol: 1e-1')
%         if isequal((absMat <= 1e-1), oneMat)
%             disp(['OK - ',srcFiles(i).name])
%         else
%             disp(['NG - ',srcFiles(i).name])
%             %is_same = false;
%         end
% %        if isequal(ismembertol(srcImg,dstImg,1e-1), oneMat)
% %            disp(['OK - ',srcFiles(i).name])
% %        else
% %            disp(['NG - ',srcFiles(i).name])
% %            is_same = false;
% %        end
    end

    if is_same
        disp('Total: OK')
    else
        disp('Total: NG')
    end

end

