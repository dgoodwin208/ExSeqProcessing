
% Calculating Correspondences in CUDA
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

function calcCorrespondencesCUDA(moving_run)

%profile on;

loadParameters;

if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
else
    filename_root = sprintf('%s_',params.FILE_BASENAME);
end

fprintf('CalcCorrespondencesInCUDA ON MOVING: %i, FIXED: %i\n', moving_run, params.REFERENCE_ROUND_WARP);


if isfield(params,'CALC_CORR_MAX_THREADS')
    maxNumCompThreads(params.CALC_CORR_MAX_THREADS);
end


filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
    filename_root,moving_run,regparams.CHANNELS{1},params.IMAGE_EXT));

try
    imgMoving_total = load3DImage_uint16(filename);
catch
    fprintf('ERROR: Cannot load file. TODO: add skippable rounds\n');
    return;
end
ymin = 1;
ymax = size(imgMoving_total,1);
xmin = 1;
xmax = size(imgMoving_total,2);


%------------------------------Load Descriptors -------------------------%
%Load all descriptors for the MOVING channel
tic;
keys_moving_sift.pos = [];
keys_moving_sift.ivec = [];
for register_channel = [regparams.REGISTERCHANNELS_SIFT]
    descriptor_output_dir_moving = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        moving_run,register_channel{1}));

    filename = fullfile(descriptor_output_dir_moving, ...
        [num2str(ymin) '-' num2str(ymax) '_' num2str(xmin) '-' num2str(xmax) '.mat']);

    data = load(filename);
    keys = vertcat(data.keys{:});
    pos = [[keys(:).y]',[keys(:).x]',[keys(:).z]'];
    ivec = vertcat(keys(:).ivec);

    keys_moving_sift.pos  = vertcat(keys_moving_sift.pos,pos);
    keys_moving_sift.ivec = vertcat(keys_moving_sift.ivec,ivec);
end
fprintf('load sift keys of moving round%03d. ',moving_run);toc;

tic;
keys_moving_sc.pos = [];
for register_channel = [regparams.REGISTERCHANNELS_SC]
    descriptor_output_dir_moving = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        moving_run,register_channel{1}));

    filename = fullfile(descriptor_output_dir_moving, ...
        [num2str(ymin) '-' num2str(ymax) '_' num2str(xmin) '-' num2str(xmax) '.mat']);

    data = load(filename);
    keys = vertcat(data.keys{:});
    pos = [[keys(:).y]',[keys(:).x]',[keys(:).z]'];

    keys_moving_sc.pos = vertcat(keys_moving_sc.pos,pos);
end
fprintf('load sc keys of moving round%03d. ',moving_run);toc;
%------------All descriptors are now loaded as keys_*_total -------------%


%loop over all the subsections desired for the piecewise affine, finding
%all relevant keypoints then calculating the transform from there
keyM_total = [];
keyF_total = [];

%Because it takes up to hours to generate the global list of vetted
%keys, after we generate them we now save them in the output_keys_filename
%if it's aready been generated, we can skip directly to the TPS calculation
output_keys_filename = fullfile(params.registeredImagesDir,sprintf('globalkeys_%sround%03d.mat',filename_root,moving_run));

%If we need to run the robust model checking to identify correct
%correspondences
if ~exist(output_keys_filename,'file')

    filename = fullfile(params.registeredImagesDir,sprintf('%sround%03d_lf_sift.mat',...
        filename_root,params.REFERENCE_ROUND_WARP));
    if (~exist(filename))
        fprintf('ShapeContext of fixed image is not calculated.\n');
        exit
    end
    load(filename);
    % 'LF_SIFT','DF_SIFT_norm','DF_SC','imgFixed_total_size','num_keys_fixed'


    num_keys_moving = length(keys_moving_sift)+length(keys_moving_sc);
    disp(['Sees ' num2str(num_keys_fixed) ' features for fixed and ' num2str(num_keys_moving) ' features for moving.']);
    if num_keys_fixed==0 || num_keys_moving==0
        disp('Empty set of descriptors. Skipping')
        return;
    end

    % ----------- SIFT MATCHING AND ROBUST MODEL SELECTION ----------%
    %

    tic;
    %Extract the keypoints-only for the shape context calculation
    %D is for descriptor, M is for moving
    DM_SIFT = keys_moving_sift.ivec;
    LM_SIFT = keys_moving_sift.pos;
    LM_SC = keys_moving_sc.pos;
    fprintf('prepare keypoints of moving round. ');toc;

    fprintf('calculating SIFT correspondences...\n');
    DM_SIFT = double(DM_SIFT);
    DM_SIFT_norm= DM_SIFT ./ repmat(sum(DM_SIFT,2),1,size(DM_SIFT,2));
    clear DM_SIFT;
    size(DM_SIFT_norm)

    tic;
    correspondences_sift = match_3DSIFTdescriptors_cuda(DM_SIFT_norm,DF_SIFT_norm);
    toc;

    fprintf('calculating ShapeContext descriptors...\n');
    %We create a shape context descriptor for the same keypoint
    %that has the SIFT descriptor.
    %So we calculate the SIFT descriptor on the normed channel
    %(summedNorm), and we calculate the Shape Context descriptor
    %using keypoints from all other channels
    DM_SC=ShapeContext(LM_SIFT,LM_SC);

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

    if length(correspondences)<20
        disp(['We only see ' num2str(length(correspondences)) ' which is insufficient to calculate a reliable transform. Skipping']);
        error('Insufficient points after filtering. Try increasing the inlier parameters in calc_affine');
        return;
    end

    LM = LM_SIFT;
    LF = LF_SIFT;
    %RANSAC filtering producing keyM and keyF varibles
    warning('off','all'); tic;calc_affine; toc;warning('on','all')

    %calc_affine produces keyM and keyF, pairs of point correspondences
    %from the robust model fitting. The math is done with local
    %coordinates to the subvolume, so it needs to be adapted to global
    %points

    keyM_total = keyM;
    keyF_total = keyF;


    clear LM_SIFT DM_SIFT_norm DM_SC LF_SIFT;
    % ----------- END ---------- %

    save(output_keys_filename,'keyM_total','keyF_total');
else %if we'va already calculated keyM_total and keyF_total, we can just load it
    disp('KeyM_total and KeyF_total already calculated. Skipping');

end

%profile off; profsave(profile('info'),sprintf('profile-results-register-with-desc-%d',moving_run));

end


