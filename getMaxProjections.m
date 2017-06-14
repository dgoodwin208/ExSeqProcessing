% Quick script to get a maxProjection of all images to show
% the difference between pre/post registration and color correction
function getMaxProjections(channels)
loadParameters;

if ~exist(params.reportingDir,'dir')
    mkdir(params.reportingDir);
end

for roundnum = 1:params.NUM_ROUNDS
    
    for channel_idx = 1:params.NUM_CHANNELS
        channel_suffix = channels{channel_idx};
        filename = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,roundnum,channel_suffix));
        fprintf('%s\n',filename);
        chanel_data = load3DTif(filename);
        
        channel_max = max(channel_data,[],3);
        save3DTif(channel_max,fullfile(params.reportingDir,sprintf('MAXPROJ_%s_round%.03i_%s.tif',params.FILE_BASENAME,roundnum,channel_suffix)));
    end

end
end