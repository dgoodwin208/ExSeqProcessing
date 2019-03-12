% Step 2-3: Keypoints and Descriptors
% Calculate the SIFT Descriptors for a subset of subvolumes of the experiments
% as dictated in the loadExperimentParams.m script.
%
% INPUTS:
% sample_num is the index of the tissue sample to register
% run_num is the index of the experiment for the specified sample
% start_idx and end_idx specify the indices of subvolumes to calculate
%        keypoints and descriptors
%
% OUTPUTS:
% The output is a file that looks like <ymin>_<ymax>_<xmin>_<xmax>.m describing
% the pixel indexs of the subregion (example: 1150-1724_1-546.mat). These files
% contain the keypoints and descriptors in the keys cell array.
%
% These files are then later loaded by the registerWithDescriptors.m file
% Author: Daniel Goodwin dgoodwin208@gmail.com
% Date: August 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function calculateDescriptors(run_num)
% CALCULATEDESCRIPTORFORTILEATINDICES  Calculates keypoints and descriptors

%Load all the parameters per file

loadParameters;

if isfield(params,'CALC_DESC_MAX_THREADS')
    maxNumCompThreads(params.CALC_DESC_MAX_THREADS);
end

if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
else
    filename_root = sprintf('%s_',params.FILE_BASENAME);
end


for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC]) 

    regChan = register_channel{1}; 
    %Loading the image file (tif/hdf5) associated with the reference channel (ie,
    %Lectin) for the image specified by run_num
    %The {1} to register_cahnnel is a gross bit of cell matlab code
    filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
        filename_root,run_num,regChan,params.IMAGE_EXT));
    img = load3DImage_uint16(filename);


    %Make sure the folders for the descriptor outputs exist:
    descriptor_output_dir = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root,run_num,register_channel{1}));
    if exist(descriptor_output_dir,'dir')==0
        mkdir(descriptor_output_dir);
    end

    % get region, indexing column-wise
    ymin = 1;
    ymax = size(img,1);
    xmin = 1;
    xmax = size(img,2);

    outputfilename = fullfile(descriptor_output_dir, ...
        [num2str(ymin) '-' num2str(ymax) '_' num2str(xmin) '-' num2str(xmax) '.mat']);

    %If we're calculating a channel just for shape context, then we
    %only need the keypoint and not the descriptor. So we do a check for any channel that
    %is only in the REGISTERCHANNELS_SC and not in
    %REGISTERCHANNELS_SIFT

    skipDescriptor = ~any(strcmp(regparams.REGISTERCHANNELS_SIFT,regChan));            
    if exist(outputfilename,'file')>0 %Make sure that the descriptors have been calculated!
        fprintf('Sees that the file %s already exists, skipping\n',outputfilename);
        return;
    else
        keys = SWITCH_tile_processingInParallel(img,skipDescriptor);
    end
    num_keys = length(keys);
    fprintf('%i SIFT-keypoints\n',length(keys));

    %There is a different terminology for the x,y coordinates that
    %needs to be noted. The 3DSIFT code treats x as the 0th
    %index (ie, the vertical dimension) whereas I am used to saying y
    %is the vertical direction.  This next bit of code both switches
    %the x and y back to my convention

    for key_idx=1:length(keys)
        tmp = keys{key_idx}.x;
        keys{key_idx}.x = keys{key_idx}.y;
        keys{key_idx}.y = tmp;
    end

    save(outputfilename,'keys','ymin','xmin','ymax','xmax', 'params','run_num');

    clear keys;

    nkeys_filename = fullfile(params.registeredImagesDir,sprintf('nkeys_%sround%03d_%s.mat',...
        filename_root,run_num,regChan));
    save(nkeys_filename,'num_keys');
end

