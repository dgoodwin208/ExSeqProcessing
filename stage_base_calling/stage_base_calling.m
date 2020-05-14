function stage_base_calling()

    loadParameters;

    [ret,messages] = check_files_in_base_calling();
    if ret
        fprintf('already processed puncta extraction\n');
        fprintf('[DONE]\n');
        return
    end

    %TODO: include base calling confidence etc. into this function
    %    currently this minimal
    if params.ISILLUMINA
        process_punctavoxels_to_transcripts_illumina
    else
        processing_targetedExSeq_fromPunctaSOLiD;
    end

    [ret,messages] = check_files_in_base_calling();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end
