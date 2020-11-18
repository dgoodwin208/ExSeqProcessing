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

    punctafeinder;
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

