function [keys] = loadFOVKeyptsAndFeatures(FOV,round,bigparams)
% Load any Fov, make it global and return


foldername = fullfile(bigparams.EXPERIMENT_FOLDERROOT,...
    sprintf('F%.3i',FOV),...
    '4_registration',...
    sprintf('%s-F%.3i-downsample_round%.3i_%s',...
    bigparams.EXPERIMENT_NAME,FOV,round,bigparams.REG_CHANNEL) );

keys = {};
if ~exist(foldername,'dir')
    return
end
%Get the file from inside the folder
files = dir(fullfile(foldername,'*.mat'));
if isempty(files)
%     fprintf('%s is empty\n',foldername);
    return
end
filename = files(1).name;

%Load the variable 'keys'
data = load(fullfile(foldername,filename));
keys = data.keys;

[row,col] = find(bigparams.TILE_MAP==FOV);

%Copy those keys into a global holder of all
%keypoints+descriptors
% fprintf('Adding %i entries from FOV %i\n',length(keys),FOV);
for k = 1:length(keys)
    
    %The position of the keypoints is in
    %downsampled coordinatees
    
    keys{k}.x_global = bigparams.IMG_SIZE_XY(1)*(col-1) + keys{k}.x*bigparams.DOWNSAMPLE_RATE;
    keys{k}.y_global = bigparams.IMG_SIZE_XY(2)*(row-1) + keys{k}.y*bigparams.DOWNSAMPLE_RATE;
    keys{k}.z_global = keys{k}.z*bigparams.DOWNSAMPLE_RATE;
    keys{k}.F = FOV;
    
    keys{k} = rmfield(keys{k},'xyScale');
    keys{k} = rmfield(keys{k},'tScale');
    keys{k} = rmfield(keys{k},'k');
end







