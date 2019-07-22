function stage_base_calling()

    loadParameters;

    %TODO: Finish implementing this, currently just a shell copied from puncta_extraction
    [ret,messages] = check_files_in_base_calling();
    if ret
        fprintf('already processed puncta extraction\n');
        fprintf('[DONE]\n');
        return
    end

    process_punctavoxels_to_transcripts;

    [ret,messages] = check_files_in_base_calling();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end

