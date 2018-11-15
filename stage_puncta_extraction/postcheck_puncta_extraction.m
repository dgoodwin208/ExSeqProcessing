function ret = postcheck_puncta_extraction()

    loadParameters;

    filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
    if ~exist(filename_out,'file')
        fprintf('[ERROR] not created: %s\n',filename_out);
        ret = false;
        return
    end

    filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
    if ~exist(filename_centroids,'file')
        fprintf('[ERROR] not created: %s\n',filename_centroids);
        ret = false;
        return
    end

    outputfile = fullfile(params.basecallingResultsDir,sprintf('%s_puncta_pixels.mat',params.FILE_BASENAME));
    if ~exist(outputfile,'file')
        fprintf('[ERROR] not created: %s\n',outputfile);
        ret = false;
        return
    end

    fprintf('[DONE]\n');

    ret = true;
end

