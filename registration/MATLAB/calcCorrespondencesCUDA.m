
% Calculating Correspondences in CUDA
% This is the code that calculates the keypoints and descriptors at
% varying scale levels
%
% INPUTS:
% moving_run: which expeirment do you want to warp accordingly?
% OUTPUTS:
% no variables. All outputs saved to regparams.OUTPUTDIR
%
% Author: Daniel Goodwin dgoodwin208@gmail.com
% Date: August 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function calcCorrespondencesCUDA(moving_run)

%profile on;

semaphore('/gr','open',1); % it is no effective if the semaphore is already open

loadParameters;

if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
else
    filename_root = sprintf('%s_',params.FILE_BASENAME);
end

fprintf('CalcCorrespondencesInCUDA ON MOVING: %i, FIXED: %i\n', moving_run, regparams.FIXED_RUN);


maxNumCompThreads(params.REG_DESC_MAX_THREADS);


filename = fullfile(regparams.INPUTDIR,sprintf('%sround%03d_%s.tif',...
    filename_root,moving_run,regparams.CHANNELS{1}));

try
    imgMoving_total = load3DTif_uint16(filename);
catch
    fprintf('ERROR: Cannot load file. TODO: add skippable rounds\n');
    return;
end


%------------------------------Load Descriptors -------------------------%
%Load all descriptors for the MOVING channel
tic;
keys_moving_total_sift.pos = [];
keys_moving_total_sift.ivec = [];
for register_channel = [regparams.REGISTERCHANNELS_SIFT]
    descriptor_output_dir_moving = fullfile(regparams.OUTPUTDIR,sprintf('%sround%03d_%s/',filename_root, ...
        moving_run,register_channel{1}));
    
    files = dir(fullfile(descriptor_output_dir_moving,'*.mat'));
    
    for file_idx= 1:length(files)
        filename = files(file_idx).name;
        
        %The data for each tile is keys, xmin, xmax, ymin, ymax
        data = load(fullfile(descriptor_output_dir_moving,filename));
        keys = vertcat(data.keys{:});
        pos = [[keys(:).y]'+data.ymin-1,[keys(:).x]'+data.xmin-1,[keys(:).z]'];
        ivec = vertcat(keys(:).ivec);

        keys_moving_total_sift.pos  = vertcat(keys_moving_total_sift.pos,pos);
        keys_moving_total_sift.ivec = vertcat(keys_moving_total_sift.ivec,ivec);
    end
end
fprintf('load sift keys of moving round%03d. ',moving_run);toc;

tic;
keys_moving_total_sc.pos = [];
for register_channel = [regparams.REGISTERCHANNELS_SC]
    descriptor_output_dir_moving = fullfile(regparams.OUTPUTDIR,sprintf('%sround%03d_%s/',filename_root, ...
        moving_run,register_channel{1}));
    
    files = dir(fullfile(descriptor_output_dir_moving,'*.mat'));
    
    for file_idx= 1:length(files)
        filename = files(file_idx).name;
        
        %The data for each tile is keys, xmin, xmax, ymin, ymax
        data = load(fullfile(descriptor_output_dir_moving,filename));
        keys = vertcat(data.keys{:});
        pos = [[keys(:).y]'+data.ymin-1,[keys(:).x]'+data.xmin-1,[keys(:).z]'];

        keys_moving_total_sc.pos = vertcat(keys_moving_total_sc.pos,pos);
    end
end
fprintf('load sc keys of moving round%03d. ',moving_run);toc;
%------------All descriptors are now loaded as keys_*_total -------------%



%chop the image up into grid
tile_upperleft_y_moving = floor(linspace(1,size(imgMoving_total,1),regparams.ROWS_TFORM+1));
tile_upperleft_x_moving = floor(linspace(1,size(imgMoving_total,2),regparams.COLS_TFORM+1));

%loop over all the subsections desired for the piecewise affine, finding
%all relevant keypoints then calculating the transform from there
keyM_total = [];
keyF_total = [];

%Because it takes up to hours to generate the global list of vetted
%keys, after we generate them we now save them in the output_keys_filename
%if it's aready been generated, we can skip directly to the TPS calculation
output_keys_filename = fullfile(regparams.OUTPUTDIR,sprintf('globalkeys_%sround%03d.mat',filename_root,moving_run));

%If we need to run the robust model checking to identify correct
%correspondences
if ~exist(output_keys_filename,'file')
    
    for x_idx=1:regparams.COLS_TFORM
        for y_idx=1:regparams.ROWS_TFORM
            
            disp(['Running on row ' num2str(y_idx) ' and col ' num2str(x_idx) ]);
            
%            tic;
            %the moving code is defined the linspace layout of dimensions above
            ymin_moving = tile_upperleft_y_moving(y_idx);
            ymax_moving = tile_upperleft_y_moving(y_idx+1);
            xmin_moving = tile_upperleft_x_moving(x_idx);
            xmax_moving = tile_upperleft_x_moving(x_idx+1);
            
            tile_img_moving = imgMoving_total(ymin_moving:ymax_moving, xmin_moving:xmax_moving,:);
            
            %Before calculating any features, make sure the tile is not empty
            if checkIfTileEmpty(tile_img_moving,regparams.EMPTY_TILE_THRESHOLD)
                disp('Sees the moving tile to be empty');
                continue
            end
            
            %FindRelevant keys not only finds the total keypoints, but converts
            %those keypoints to the scope of the specific tile, not the global
            %position
%            keys_moving = findRelevantKeys(keys_moving_total, ymin_moving, ymax_moving,xmin_moving,xmax_moving);
%            fprintf('findRelevantKeys. keys_moving(orig) ');toc;

            tic;
            keys_moving_sift_index = find(keys_moving_total_sift.pos(:,1)>=ymin_moving & keys_moving_total_sift.pos(:,1)<=ymax_moving & ...
                keys_moving_total_sift.pos(:,2)>=xmin_moving & keys_moving_total_sift.pos(:,2)<=xmax_moving);
            keys_moving_sift.pos = keys_moving_total_sift.pos(keys_moving_sift_index,:)-[ymin_moving-1,xmin_moving-1,0];
            keys_moving_sift.ivec = keys_moving_total_sift.ivec(keys_moving_sift_index,:);
            keys_moving_sc_index = find(keys_moving_total_sc.pos(:,1)>=ymin_moving & keys_moving_total_sc.pos(:,1)<=ymax_moving & ...
                keys_moving_total_sc.pos(:,2)>=xmin_moving & keys_moving_total_sc.pos(:,2)<=xmax_moving);
            keys_moving_sc.pos = keys_moving_total_sc.pos(keys_moving_sc_index,:)-[ymin_moving-1,xmin_moving-1,0];
            fprintf('findRelevantKeys. keys_moving(mod) ');toc;


            filename = fullfile(regparams.OUTPUTDIR,sprintf('%sround%03d_lf_sift_r%uc%u.mat',...
                filename_root,regparams.FIXED_RUN,y_idx,x_idx));
            if (~exist(filename))
                fprintf('ShapeContext of fixed image is not calculated.\n');
                exit
            end
            load(filename);
            % 'LF_SIFT','DF_SIFT_norm','DF_SC','imgFixed_total_size','num_keys_fixed','ymin_fixed','xmin_fixed'


            num_keys_moving = length(keys_moving_sift)+length(keys_moving_sc);
            disp(['Sees ' num2str(num_keys_fixed) ' features for fixed and ' num2str(num_keys_moving) ' features for moving.']);
            if num_keys_fixed==0 || num_keys_moving==0
                disp('Empty set of descriptors. Skipping')
                continue;
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
            
            %REMOVED: it cost a lot in time and removed only a handful of
            %points out of tens of thousands.
%            tic;
%            % deduplicate any SIFT keypoints
%            fprintf('(%i) before dedupe, ',size(LM_SIFT,1));
%            [LM_SIFT,keepM,~] = unique(LM_SIFT,'rows');
%            DM_SIFT = DM_SIFT(keepM,:);
%            fprintf('(%i) after dedupe\n',size(LM_SIFT,1));
%            
%            % deuplicate any ShapeContext keypoints
%            fprintf('(%i) before dedupe, ',size(LM_SC,1));
%            [LM_SC,~,~] = unique(LM_SC,'rows');
%            fprintf('(%i) after dedupe\n',size(LM_SC,1));
%            fprintf('deduplicate any SIFT keypoints. ');toc;
            
            
            fprintf('calculating SIFT correspondences...\n');
            DM_SIFT = double(DM_SIFT);
            DM_SIFT_norm= DM_SIFT ./ repmat(sum(DM_SIFT,2),1,size(DM_SIFT,2));
            clear DM_SIFT;
            size(DM_SIFT_norm)

            while true
                ret = semaphore('/gr','trywait');
                if ret == 0
                    break;
                else
                    pause(1);
                end
            end
            tic;
            correspondences_sift = match_3DSIFTdescriptors_cuda(DM_SIFT_norm,DF_SIFT_norm);
            toc;
            ret = semaphore('/gr','post');
            
            fprintf('calculating ShapeContext descriptors...\n');
            %We create a shape context descriptor for the same keypoint
            %that has the SIFT descriptor.
            %So we calculate the SIFT descriptor on the normed channel
            %(summedNorm), and we calculate the Shape Context descriptor
            %using keypoints from all other channels
            DM_SC=ShapeContext(LM_SIFT,LM_SC);

%            while true
%                ret = semaphore('/gr','trywait');
%                if ret == 0
%                    break;
%                else
%                    pause(1);
%                end
%            end
%            tic;
%            fprintf('calculating ShapeContext correspondences...\n');
%            correspondences_sc = match_3DSIFTdescriptors_cuda(DM_SC',DF_SC');
%            toc;
%            ret = semaphore('/gr','post');
%
%            fprintf('SIFT-only correspondences get %i matches, SC-only gets %i matches\n',...
%                size(correspondences_sift,2),size(correspondences_sc,2));
%
%            correspondences_combine = [correspondences_sc,correspondences_sift]';
%            [correspondences,~,~] = unique(correspondences_combine,'rows');
%            correspondences = correspondences';
%            fprintf('There unique %i matches if we take the union of the two methods\n', size(correspondences,2));
%
%            while true
%                ret = semaphore('/gr','trywait');
%                if ret == 0
%                    break;
%                else
%                    pause(1);
%                end
%            end
%            fprintf('calculating SIFT+ShapeContext correspondences...\n');
%            tic;
%            correspondences = match_3DSIFTdescriptors_cuda([DM_SC; DM_SIFT_norm']',[DF_SC; DF_SIFT_norm']');
%            toc;
%            ret = semaphore('/gr','post');

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
                continue;
            end
            
            LM = LM_SIFT;
            LF = LF_SIFT;
            %RANSAC filtering producing keyM and keyF varibles
            warning('off','all'); tic;calc_affine; toc;warning('on','all')
            
            %calc_affine produces keyM and keyF, pairs of point correspondences
            %from the robust model fitting. The math is done with local
            %coordinates to the subvolume, so it needs to be adapted to global
            %points
            
            keyM_total = [keyM_total; keyM(:,1) + ymin_moving-1, keyM(:,2) + xmin_moving-1, keyM(:,3) ];
            keyF_total = [keyF_total; keyF(:,1) + ymin_fixed-1, keyF(:,2) + xmin_fixed-1, keyF(:,3)];
            
            
            clear LM_SIFT DM_SIFT_norm DM_SC LF_SIFT;
            % ----------- END ---------- %
        end
    end
    
    save(output_keys_filename,'keyM_total','keyF_total');
else %if we'va already calculated keyM_total and keyF_total, we can just load it
    disp('KeyM_total and KeyF_total already calculated. Skipping');
    
end

%profile off; profsave(profile('info'),sprintf('profile-results-register-with-desc-%d',moving_run));

end


