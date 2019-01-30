function stage_puncta_extraction()

    loadParameters;

    [ret,messages] = check_files_in_puncta_extraction();
    if ret
        fprintf('already processed puncta extraction\n');
        fprintf('[DONE]\n');
        return
    end

    punctafeinder;
    clearvars

    puncta_roicollect_bgincl;

    [ret,messages] = check_files_in_puncta_extraction();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end

