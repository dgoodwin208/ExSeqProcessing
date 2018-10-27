% Quick script to get a maxProjection of all images to show
% the difference between pre/post registration and color correction
function reporting_getMaxProjections(directory_images, channels)
loadParameters;

if ~exist(params.reportingDir,'dir')
    mkdir(params.reportingDir);
end

for roundnum = 1:params.NUM_ROUNDS
    
    for channel_idx = 1:length(channels)
        channel_suffix = channels{channel_idx};
        filename = fullfile(directory_images,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,roundnum,channel_suffix));
        if ~exist(filename)
             fprintf('%s is not found, skipping\n',filename);
             continue;
        end

       filename_output = fullfile(params.reportingDir,sprintf('MAXPROJ_%s_round%.03i_%s.tif',params.FILE_BASENAME,roundnum,channel_suffix));
       if exist(filename_output)
             fprintf('%s is already there!, skipping\n',filename);
             continue;
        end 
        
        fprintf('Making max projection %s\n', filename_output);
        channel_data = load3DTif_uint16(filename);
        
        channel_max = max(channel_data,[],3);
        save3DTif_uint16(channel_max,filename_output);
    end

end
end
