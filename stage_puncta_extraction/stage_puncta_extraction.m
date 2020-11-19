function stage_puncta_extraction()

    loadParameters;
    %Save the loadParameters into the output director in case we need to do
    %later debugging
    copyfile('loadParameters.m',fullfile(params.punctaSubvolumeDir,...
        sprintf('loadParameters.m.log-%s',date)))
    [ret,messages] = check_files_in_puncta_extraction();
    if ret
        fprintf('already processed puncta extraction\n');
        fprintf('[DONE]\n');
        return
    end

    %Because we need to reprocess some fields of view, try using this
    puncta_img_mask_file = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
    if ~exist(puncta_img_mask_file,'file')
        punctafeinder;
    end
    clearvars

    %Produce a punctavoxels.mat file
    puncta_roicollect;

    [ret,messages] = check_files_in_puncta_extraction();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end

