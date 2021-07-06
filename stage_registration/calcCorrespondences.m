
% Calculating Correspondences
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

function calcCorrespondences(moving_run)

loadParameters;

if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
else
    filename_root = sprintf('%s_',params.FILE_BASENAME);
end

if isfield(params,'CALC_CORR_MAX_THREADS')
    maxNumCompThreads(params.CALC_CORR_MAX_THREADS);
end

fprintf('CalcCorrespondences ON MOVING: %i, FIXED: %i\n', moving_run, params.REFERENCE_ROUND_WARP);


filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
    filename_root,moving_run,regparams.CHANNELS{1},params.IMAGE_EXT));


img_total_size = image_dimensions(filename);
ymin = 1;
ymax = img_total_size(1);
xmin = 1;
xmax = img_total_size(2);


%------------------------------Load Descriptors -------------------------%
%Load all descriptors for the MOVING channel
keys_moving = {}; keys_ctr=1;
for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
    descriptor_output_dir_moving = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        moving_run,register_channel{1}));

    
    filename = fullfile(descriptor_output_dir_moving, ...
        [num2str(xmin) '-' num2str(xmax) '_' num2str(ymin) '-' num2str(ymax) '.mat']);

    data = load(filename);
    for idx=1:length(data.keys)
        %copy all the keys into one large vector of cells
        keys_moving{keys_ctr} = data.keys{idx};
        keys_moving{keys_ctr}.x = data.keys{idx}.x;
        keys_moving{keys_ctr}.y = data.keys{idx}.y;
        keys_moving{keys_ctr}.channel = register_channel;
        keys_ctr = keys_ctr+ 1;
    end
end

%Load all descriptors for the FIXED channel
keys_fixed = {}; keys_ctr=1;
for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
    descriptor_output_dir_fixed = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        params.REFERENCE_ROUND_WARP,register_channel{1}));

     
    %Fixed old code: since we only save one file for the descriptors in a given round
    %We can simply load that one file here. In an old version of the code, we would subsegment
    %the descriptor calculation, which is why we had coordinates in the .mat file. -DG 20200601
    files = dir(fullfile(descriptor_output_dir_fixed,'*.mat'));
    filename = fullfile(files(1).folder,files(1).name);
    
    data = load(filename);
    for idx=1:length(data.keys)
        %copy all the keys into one large vector of cells
        keys_fixed{keys_ctr} = data.keys{idx};        %#ok<*AGROW>
        keys_fixed{keys_ctr}.x = data.keys{idx}.x;
        keys_fixed{keys_ctr}.y = data.keys{idx}.y;
        keys_fixed{keys_ctr}.channel = register_channel;

        keys_ctr = keys_ctr+ 1;
    end
end
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
    LM_SC = [];
    for i = 1:length(keys_moving)
        %If this channel is to be included in the SIFT_registration
        if any(strcmp(regparams.REGISTERCHANNELS_SIFT,keys_moving{i}.channel))
            DM_SIFT(ctr_sift,:) = keys_moving{i}.ivec;
            LM_SIFT(ctr_sift,:) = [keys_moving{i}.y, keys_moving{i}.x, keys_moving{i}.z];
            ctr_sift = ctr_sift+1;
        end

        if any(strcmp(regparams.REGISTERCHANNELS_SC,keys_moving{i}.channel))
            LM_SC(ctr_sc,:) = [keys_moving{i}.y, keys_moving{i}.x, keys_moving{i}.z];
            ctr_sc = ctr_sc+1;
        end

    end

    %F for fixed
    DF_SIFT = [];
    LF_SIFT = []; ctr_sift = 1; ctr_sc = 1;
    LF_SC = [];
    for i = 1:length(keys_fixed)
        if any(strcmp(regparams.REGISTERCHANNELS_SIFT,keys_fixed{i}.channel))
            DF_SIFT(ctr_sift,:) = keys_fixed{i}.ivec;
            LF_SIFT(ctr_sift,:) = [keys_fixed{i}.y, keys_fixed{i}.x, keys_fixed{i}.z];
            ctr_sift = ctr_sift+1;
        end
        
        if any(strcmp(regparams.REGISTERCHANNELS_SC,keys_fixed{i}.channel))
            LF_SC(ctr_sc,:) = [keys_fixed{i}.y, keys_fixed{i}.x, keys_fixed{i}.z];
            ctr_sc = ctr_sc+1;
        end
    end

    DM_SIFT_norm= DM_SIFT ./ repmat(sum(DM_SIFT,2),1,size(DM_SIFT,2));
    DF_SIFT_norm= DF_SIFT ./ repmat(sum(DF_SIFT,2),1,size(DF_SIFT,2));
    %correspondences_sift = vl_ubcmatch(DM_SIFT_norm',DF_SIFT_norm');
    correspondences_sift = match_3DSIFTdescriptors(DM_SIFT_norm,DF_SIFT_norm);

    %We create a shape context descriptor for the same keypoint
    %that has the SIFT descriptor.
    %So we calculate the SIFT descriptor on the normed channel
    %(summedNorm), and we calculate the Shape Context descriptor
    %using keypoints from all other channels
    % NOTE: ShapeContext has been deprecated for now, -DG 2020-09-09
    %DM_SC=ShapeContext(LM_SIFT,LM_SC);
    %DF_SC=ShapeContext(LF_SIFT,LF_SC);

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
    warning('off','all'); tic;calc_affine; toc;warning('on','all')

    %calc_affine produces keyM and keyF, pairs of point correspondences
    %from the robust model fitting. The math is done with local
    %coordinates to the subvolume, so it needs to be adapted to global
    %points

    keyM_total = keyM;
    keyF_total = keyF;


    % ----------- END ---------- %
 


    save(output_keys_filename,'keyM_total','keyF_total');
else %if we'va already calculated keyM_total and keyF_total, we can just load it
    disp('KeyM_total and KeyF_total already calculated. Skipping');
end

end


