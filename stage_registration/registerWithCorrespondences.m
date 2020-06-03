
% Step 4-6: Calculating Warps using registration infromation
% This is the code that calculates the keypoints and descriptors at
% varying scale levels
%
% INPUTS:
% moving_run: which expeirment do you want to warp accordingly?
% OUTPUTS:
% no variables. All outputs saved to params.registeredImagesDir
%
% Author: Daniel Goodwin dgoodwin208@gmail.com
% Date: August 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function registerWithCorrespondences(moving_run,do_downsample)

loadParameters;

if do_downsample
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
end

fprintf('RegWithCorr ON MOVING: %i, FIXED: %i\n', moving_run, params.REFERENCE_ROUND_WARP);
output_affine_filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.%s',filename_root,moving_run,regparams.CHANNELS{end},params.IMAGE_EXT));
if exist(output_affine_filename,'file')
    fprintf('Already sees the last output file, skipping!\n');
    return;
end

if isfield(params,'REG_CORR_MAX_THREADS')
    maxNumCompThreads(params.REG_CORR_MAX_THREADS);
end

%Load a full-res image 
filename = fullfile(params.normalizedImagesDir,sprintf('%s_round%03d_%s.%s',...
    filename_root,params.REFERENCE_ROUND_WARP,regparams.CHANNELS{1},params.IMAGE_EXT ));
imgFixed_total = load3DImage_uint16(filename);
%Loading the keys, possibly from the downsampled data
output_keys_filename = fullfile(params.registeredImagesDir,sprintf('globalkeys_%s-downsample_round%03d.mat',params.FILE_BASENAME,moving_run));

%The globalkeys file loads the keyM and keyF matrices that are used to
%calculate the warps
load(output_keys_filename);

%This is quick hack as we figure out how to design the interface for when to apply the downsample. 
%If we're applying the warp that was calcualted from downsampled data, but
%we're applying it to the full-res data
if ~do_downsample && params.DO_DOWNSAMPLE
   % Scale the points of correspondence pairs back into original size
   keyM_total = keyM_total*params.DOWNSAMPLE_RATE;
   keyF_total = keyF_total*params.DOWNSAMPLE_RATE;
end

%First we do do a global affine transform on the data and keypoints before
%doing the fine-resolution non-rigid warp

%Because of the annoying switching between XY/YX conventions,
%we have to switch XY components for the affine calcs, then switch back
keyM_total_switch = keyM_total(:,[2 1 3]);
keyF_total_switch = keyF_total(:,[2 1 3]);

%The old way was calculating the affine tform
warning('off','all'); 
affine_tform = findAffineModel(keyM_total_switch, keyF_total_switch,regparams.AFFINE_FULL)
warning('on','all')

%Warp the keyM features into the new space
rF = imref3d(size(imgFixed_total));
%Key total_affine is now with the switched XY
keyM_total_affine = [keyM_total_switch, ones(size(keyM_total_switch,1),1)]*affine_tform';
%keyM_total is now switched
keyM_total=keyM_total_affine(:,1:3);

%Save the transform for use later
output_transform_filename = fullfile(params.registeredImagesDir,sprintf('affineTForm_%s_round%03d.mat',filename_root,moving_run));
save(output_transform_filename,'affine_tform');

%Remove any keypoints which are now outside the bounds of the image
filtered_correspondence_indices = (keyM_total(:,1) <1 | keyM_total(:,2)<1 | keyM_total(:,3)<1 | ...
    keyM_total(:,1) > size(imgFixed_total,2) | ...
    keyM_total(:,2) > size(imgFixed_total,1) | ...
    keyM_total(:,3) > size(imgFixed_total,3) );
fprintf('Losing %i features after affine warp\n',sum(filtered_correspondence_indices));
keyM_total(filtered_correspondence_indices,:) = [];
keyF_total(filtered_correspondence_indices,:) = [];

%switch keyM back to the other format for the TPS calcs
keyM_total = keyM_total(:,[2 1 3]);

for c = 1:length(regparams.CHANNELS)
    %Load the data to be warped
    tic;
    data_channel = regparams.CHANNELS{c};
    fprintf('load 3D file for affine transform on %s channel\n',data_channel);
    %filename = fullfile(params.normalizedImagesDir,sprintf('%s_round%03d_%s.%s',params.FILE_BASENAME,moving_run,data_channel,params.IMAGE_EXT));
    filename = fullfile(params.normalizedImagesDir,sprintf('%s_round%03d_%s.%s',filename_root,moving_run,data_channel,params.IMAGE_EXT));
    imgToWarp = load3DImage_uint16(filename);
    toc;
    
    output_affine_filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.%s',...
        filename_root,moving_run,data_channel,params.IMAGE_EXT));
    
    imgMoving_total_affine = imwarp(imgToWarp,affine3d(affine_tform'),'OutputView',rF);
    save3DImage_uint16(imgMoving_total_affine,output_affine_filename);
end

if strcmp(regparams.REGISTRATION_TYPE,'affine')
    fprintf('Ending the registration after the affine\n');
    return;
end

%fprintf('Setting up cluster and parpool for TPS warp')
%tic;
%cluster = parcluster('local');
%parpool(cluster) %Don't check these changes, this is local hacks
%toc;

output_TPS_filename = fullfile(params.registeredImagesDir,sprintf('TPSMap_%s_round%03d.mat',filename_root,moving_run));
if exist(output_TPS_filename,'file')==0
    
    %These keypoints have now been modified by the affine warp, so are in
    %the dimensinos of the keyFTotal
    [in1D_total,out1D_total] = TPS3DWarpWholeInParallel(keyM_total,keyF_total, ...
        size(imgFixed_total), size(imgFixed_total));
    disp('save TPS file')
    tic;
    save(output_TPS_filename,'in1D_total','out1D_total','-v7.3');
    toc;
else
    %load in1D_total and out1D_total
    disp('load TPS file')
    tic;
    load(output_TPS_filename);
    toc;
    %Experiments 7 and 8 may have been saved with zeros in the 1D vectors
    %so this removes it
    [ValidIdxs,~] = find(in1D_total>0);
    in1D_total = in1D_total(ValidIdxs);
    out1D_total = out1D_total(ValidIdxs);
end

%Warp all three channels of the experiment once the index mapping has been
%created
for c = 1:length(regparams.CHANNELS)
    %Load the data to be warped
    disp('load 3D file to be warped')
    tic;
    data_channel = regparams.CHANNELS{c};
    filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.%s',filename_root,moving_run,data_channel,params.IMAGE_EXT));
    imgToWarp = load3DImage_uint16(filename);
    toc;
    
    [ outputImage_interp ] = TPS3DApply(in1D_total,out1D_total,imgToWarp,size(imgFixed_total));
    
    outputfile = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_%s.%s',filename_root,moving_run,data_channel,regparams.REGISTRATION_TYPE,params.IMAGE_EXT));
    save3DImage_uint16(outputImage_interp,outputfile);
end

disp('delete parpool')
tic;
p = gcp('nocreate');
delete(p);
toc;







end


