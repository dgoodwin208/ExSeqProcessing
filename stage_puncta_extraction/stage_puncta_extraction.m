function stage_puncta_extraction()

    loadParameters;
    %Save the loadParameters into the output director in case we need to do
    %later debugging
    copyfile('loadParameters.m',fullfile(params.punctaSubvolumeDir,...
        sprintf('loadParameters.m.log-%s',date)))
    
    [ret,messages] = check_files_in_puncta_extraction();
    
    if ret
        %IN BRANCH: 
        %Check the number of rounds that have been processed 
        punctafile = fullfile(params.punctaSubvolumeDir,sprintf('%s_punctavoxels.mat',params.FILE_BASENAME));
        load(punctafile);
        if length(puncta_set_cell)<params.NUM_ROUNDS
            fprintf('Reprocessing the ROI_collect because %i/%i rounds processed\n',...
            length(puncta_set_cell),params.NUM_ROUNDS);
        else
            fprintf('already processed puncta extraction\n');
            fprintf('[DONE]\n');
            return
        end
    end

    %IN BRANCH: 
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

