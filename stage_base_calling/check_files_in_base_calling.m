function [ret,messages] = check_files_in_base_calling()

    loadParameters;

    ret = true;
    messages = {};

    outputfile = fullfile(params.basecallingResultsDir,sprintf('%s_results.mat',params.FILE_BASENAME));
    if ~exist(outputfile,'file')
        ret = false;
        messages{end+1} = sprintf('[ERROR] not created: %s',outputfile);
        return
    end
end

