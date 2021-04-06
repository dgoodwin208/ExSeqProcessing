function stage_base_calling()

    loadParameters;
    %Save the loadParameters into the output director in case we need to do
    %later debugging
    copyfile('loadParameters.m',fullfile(params.basecallingResultsDir,...
        sprintf('loadParameters.m.log-%s',date)))
    [ret,messages] = check_files_in_base_calling();
    if ret && ~params.OVERWRITE_PREV_RESULTS
        fprintf('already processed the sequences\n');
        fprintf('[DONE]\n');
        return
    end
    
    if params.OVERWRITE_PREV_RESULTS && exist(fullfile(params.basecallingResultsDir,sprintf('%s_transcriptobjects.mat',params.FILE_BASENAME),'file')
	fprintf('Deleting the previous basecalls.\n');
	delete(fullfile(params.basecallingResultsDir,sprintf('%s_transcriptobjects.mat',params.FILE_BASENAME)) );
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
