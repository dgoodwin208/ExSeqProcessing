%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Need to update this header
% This is the code that calculates the keypoints and descriptors at
% varying scale levels
%
% INPUTS:
% moving_run: which expeirment do you want to warp accordingly?
% OUTPUTS:
% no variables. All outputs saved to params.registeredImagesDir
%
% Author: Daniel Goodwin dgoodwin208@gmail.com
% Date: December 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed)

loadParameters;

if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
else
    filename_root = sprintf('%s_',params.FILE_BASENAME);
end

if isfield(params,'CALC_CORR_MAX_THREADS')
    maxNumCompThreads(params.CALC_CORR_MAX_THREADS);
end


%loop over all the subsections desired for the piecewise affine, finding
%all relevant keypoints then calculating the transform from there
keyM_total = [];
keyF_total = [];


%If we need to run the robust model checking to identify correct
%correspondences


disp(['Sees ' num2str(length(keys_fixed)) ' features for fixed and ' num2str(length(keys_moving)) ' features for moving.']);
if length(keys_fixed)==0 || length(keys_moving)==0
    disp('Empty set of descriptors. Skipping')
    return;
end

% ----------- SIFT MATCHING AND ROBUST MODEL SELECTION ----------%
%

%Extract the keypoints-only for the shape context calculation
%D is for descriptor, M is for moving
DM_SIFT = []; %DM_SC is defined later
LM_SIFT = []; ctr_sift = 1; ctr_sc = 1;

for i = 1:length(keys_moving)
    
    DM_SIFT(ctr_sift,:) = keys_moving{i}.ivec;
    LM_SIFT(ctr_sift,:) = [keys_moving{i}.y, keys_moving{i}.x, keys_moving{i}.z];
    ctr_sift = ctr_sift+1;
end

%F for fixed
DF_SIFT = [];
LF_SIFT = []; ctr_sift = 1;

for i = 1:length(keys_fixed)
    
    DF_SIFT(ctr_sift,:) = keys_fixed{i}.ivec;
    LF_SIFT(ctr_sift,:) = [keys_fixed{i}.y, keys_fixed{i}.x, keys_fixed{i}.z];
    ctr_sift = ctr_sift+1;
    
end

DM_SIFT_norm= DM_SIFT ./ repmat(sum(DM_SIFT,2),1,size(DM_SIFT,2));
DF_SIFT_norm= DF_SIFT ./ repmat(sum(DF_SIFT,2),1,size(DF_SIFT,2));

%correspondences_sift = vl_ubcmatch(DM_SIFT_norm',DF_SIFT_norm');
correspondences_sift = match_3DSIFTdescriptors(DM_SIFT_norm,DF_SIFT_norm);

correspondences=correspondences_sift;
%Check for duplicate matches- ie, keypoint A matching to both
%keypoint B and keypoint C

num_double_matches = 0;
for idx = 1:2
    u=unique(correspondences(idx ,:));         % the unique values
    [n,~]=histc(correspondences(idx ,:),u);  % count how many of each and where
    col_duplicates_indices=find(n>1);       % index to bin w/ more than one
    num_double_matches = num_double_matches + length(col_duplicates_indices);
    correspondences(:,col_duplicates_indices) = [];
end
fprintf('There are %i matches when combining the features evenly (removed %i double matches)\n', size(correspondences,2),num_double_matches);


if length(correspondences)<regparams.NCORRESPONDENCES_MIN
    disp(['We only see ' num2str(length(correspondences)) ' which is insufficient to calculate a reliable transform. Skipping']);
    error('Insufficient points after filtering. Try increasing the inlier parameters in calc_affine');
    return;
end

LM = LM_SIFT;
LF = LF_SIFT;
%RANSAC filtering producing keyM and keyF varibles
warning('off','all'); tic;
calc_affine;
toc;warning('on','all')

%calc_affine produces keyM and keyF, pairs of point correspondences
%from the robust model fitting. The math is done with local
%coordinates to the subvolume, so it needs to be adapted to global
%points

keyM_total = keyM;
keyF_total = keyF;






