function [ret,messages] = check_files_in_puncta_extraction()

    loadParameters;

    ret = true;
    messages = {};

    filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
    if ~exist(filename_out,'file')
        ret = false;
        messages{end+1} = sprintf('[ERROR] not created: %s',filename_out);
        return
    end

    filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
    if ~exist(filename_centroids,'file')
        ret = false;
        messages{end+1} = sprintf('[ERROR] not created: %s',filename_centroids);
        return
    end

    outputfile = fullfile(params.basecallingResultsDir,sprintf('%s_puncta_pixels.mat',params.FILE_BASENAME));
    if ~exist(outputfile,'file')
        ret = false;
        messages{end+1} = sprintf('[ERROR] not created: %s',outputfile);
        return
    end
end

