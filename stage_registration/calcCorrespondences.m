
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

fprintf('CalcCorrespondences ON MOVING: %i, FIXED: %i\n', moving_run, regparams.FIXED_RUN);


filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
    filename_root,regparams.FIXED_RUN,regparams.CHANNELS{1},params.IMAGE_EXT ));

imgFixed_total = load3DImage_uint16(filename);


filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
    filename_root,moving_run,regparams.CHANNELS{1},params.IMAGE_EXT));

try
    imgMoving_total = load3DImage_uint16(filename);
catch
    fprintf('ERROR: Cannot load file. TODO: add skippable rounds\n');
    return;
end


%------------------------------Load Descriptors -------------------------%
%Load all descriptors for the MOVING channel
keys_moving_total = {}; keys_ctr=1;
for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
    descriptor_output_dir_moving = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        moving_run,register_channel{1}));
    
    files = dir(fullfile(descriptor_output_dir_moving,'*.mat'));
    
    for file_idx= 1:length(files)
        filename = files(file_idx).name;
        
        %The data for each tile is keys, xmin, xmax, ymin, ymax
        data = load(fullfile(descriptor_output_dir_moving,filename));
        for idx=1:length(data.keys)
            %copy all the keys into one large vector of cells
            keys_moving_total{keys_ctr} = data.keys{idx};
            keys_moving_total{keys_ctr}.x = data.keys{idx}.x + data.xmin-1;
            keys_moving_total{keys_ctr}.y = data.keys{idx}.y + data.ymin-1;
            keys_moving_total{keys_ctr}.channel = register_channel;
            keys_ctr = keys_ctr+ 1;
        end
    end
end

%Load all descriptors for the FIXED channel
keys_fixed_total = {}; keys_ctr=1;
for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
    descriptor_output_dir_fixed = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        regparams.FIXED_RUN,register_channel{1}));
    
    files = dir(fullfile(descriptor_output_dir_fixed,'*.mat'));
    
    for file_idx= 1:length(files)
        filename = files(file_idx).name;
        
        data = load(fullfile(descriptor_output_dir_fixed,filename));
        for idx=1:length(data.keys)
            %copy all the keys into one large vector of cells
            keys_fixed_total{keys_ctr} = data.keys{idx};        %#ok<*AGROW>
            keys_fixed_total{keys_ctr}.x = data.keys{idx}.x + data.xmin-1;
            keys_fixed_total{keys_ctr}.y = data.keys{idx}.y + data.ymin-1;
            keys_fixed_total{keys_ctr}.channel = register_channel;
            
            keys_ctr = keys_ctr+ 1;
        end
    end
end
%------------All descriptors are now loaded as keys_*_total -------------%



%chop the image up into grid
tile_upperleft_y_moving = floor(linspace(1,size(imgMoving_total,1),regparams.ROWS_TFORM+1));
tile_upperleft_x_moving = floor(linspace(1,size(imgMoving_total,2),regparams.COLS_TFORM+1));

%don't need to worry about padding because these tiles are close enough in
%(x,y) origins
tile_upperleft_y_fixed = floor(linspace(1,size(imgFixed_total,1),regparams.ROWS_TFORM+1));
tile_upperleft_x_fixed = floor(linspace(1,size(imgFixed_total,2),regparams.COLS_TFORM+1));

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
    
    for x_idx=1:regparams.COLS_TFORM
        for y_idx=1:regparams.ROWS_TFORM
            
            disp(['Running on row ' num2str(y_idx) ' and col ' num2str(x_idx) ]);
            
            %the moving code is defined the linspace layout of dimensions above
            ymin_moving = tile_upperleft_y_moving(y_idx);
            ymax_moving = tile_upperleft_y_moving(y_idx+1);
            xmin_moving = tile_upperleft_x_moving(x_idx);
            xmax_moving = tile_upperleft_x_moving(x_idx+1);
            
            tile_img_moving = imgMoving_total(ymin_moving:ymax_moving, xmin_moving:xmax_moving,:);
            
            %Before calculating any features, make sure the tile is not empty
            %if checkIfTileEmpty(tile_img_moving,regparams.EMPTY_TILE_THRESHOLD)
            %    disp('Sees the moving tile to be empty');
            %    continue
            %end
            
            %FindRelevant keys not only finds the total keypoints, but converts
            %those keypoints to the scope of the specific tile, not the global
            %position
            keys_moving = findRelevantKeys(keys_moving_total, ymin_moving, ymax_moving,xmin_moving,xmax_moving);
            
            %Loading the fixed tiles is detemined by some extra overlap between
            %the tiles (may not be necessary)
            tile_img_fixed_nopadding = imgFixed_total(tile_upperleft_y_fixed(y_idx):tile_upperleft_y_fixed(y_idx+1), ...
                tile_upperleft_x_fixed(x_idx):tile_upperleft_x_fixed(x_idx+1), ...
                :);
            tilesize_fixed = size(tile_img_fixed_nopadding);
            
            ymin_fixed = tile_upperleft_y_fixed(y_idx);
            ymax_fixed = tile_upperleft_y_fixed(y_idx+1);
            xmin_fixed = tile_upperleft_x_fixed(x_idx);
            xmax_fixed = tile_upperleft_x_fixed(x_idx+1);
            
            ymin_fixed_overlap = floor(max(tile_upperleft_y_fixed(y_idx)-(regparams.OVERLAP/2)*tilesize_fixed(1),1));
            ymax_fixed_overlap = floor(min(tile_upperleft_y_fixed(y_idx+1)+(regparams.OVERLAP/2)*tilesize_fixed(1),size(imgFixed_total,1)));
            xmin_fixed_overlap = floor(max(tile_upperleft_x_fixed(x_idx)-(regparams.OVERLAP/2)*tilesize_fixed(2),1));
            xmax_fixed_overlap = floor(min(tile_upperleft_x_fixed(x_idx+1)+(regparams.OVERLAP/2)*tilesize_fixed(2),size(imgFixed_total,2)));
            
            clear tile_img_fixed_nopadding;
            
            tile_img_fixed = imgFixed_total(ymin_fixed_overlap:ymax_fixed_overlap, xmin_fixed_overlap:xmax_fixed_overlap,:);
            
            if checkIfTileEmpty(tile_img_fixed,regparams.EMPTY_TILE_THRESHOLD)
                disp('Sees the moving tile to be empty');
                continue
            end
            
            %FindRelevant keys not only finds the total keypoints, but converts
            %those keypoints to the scope of the specific tile, not the global
            %position
            keys_fixed = findRelevantKeys(keys_fixed_total, ymin_fixed, ymax_fixed,xmin_fixed,xmax_fixed);
            
            disp(['Sees ' num2str(length(keys_fixed)) ' features for fixed and ' num2str(length(keys_moving)) ' features for moving.']);
            if length(keys_fixed)==0 || length(keys_moving)==0
                disp('Empty set of descriptors. Skipping')
                continue;
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
            
            %REMOVED: it cost a lot in time and removed only a handful of
            %points out of tens of thousands.
            
            % deuplicate any SIFT keypoints
%             fprintf('(%i,%i) before dedupe, ',size(LM_SIFT,1),size(LF_SIFT,1));
%             [LM_SIFT,keepM,~] = unique(LM_SIFT,'rows');
%             [LF_SIFT,keepF,~] = unique(LF_SIFT,'rows');
%             DM_SIFT = DM_SIFT(keepM,:);
%             DF_SIFT = DF_SIFT(keepF,:);
%             fprintf('(%i,%i) after dedupe\n',size(LM_SIFT,1),size(LF_SIFT,1));
            
            % deuplicate any ShapeContext keypoints
            %fprintf('(%i,%i) before dedupe, ',size(LM_SC,1),size(LF_SC,1));
            %[LM_SC,~,~] = unique(LM_SC,'rows');
            %[LF_SC,~,~] = unique(LF_SC,'rows');
            %fprintf('(%i,%i) after dedupe\n',size(LM_SC,1),size(LF_SC,1));
            %--end removal of deduplication code
            
            DM_SIFT_norm= DM_SIFT ./ repmat(sum(DM_SIFT,2),1,size(DM_SIFT,2));
            DF_SIFT_norm= DF_SIFT ./ repmat(sum(DF_SIFT,2),1,size(DF_SIFT,2));
            %correspondences_sift = vl_ubcmatch(DM_SIFT_norm',DF_SIFT_norm');
            correspondences_sift = match_3DSIFTdescriptors(DM_SIFT_norm,DF_SIFT_norm);
            
            %We create a shape context descriptor for the same keypoint
            %that has the SIFT descriptor.
            %So we calculate the SIFT descriptor on the normed channel
            %(summedNorm), and we calculate the Shape Context descriptor
            %using keypoints from all other channels
            
            DM_SC=ShapeContext(LM_SIFT,LM_SC);
            DF_SC=ShapeContext(LF_SIFT,LF_SC);
            %[DM_SC,DF_SC]=ShapeContext(LM_SIFT,LM_SC,LF_SIFT,LF_SC);
            
            if 0
                %correspondences_sc = vl_ubcmatch(DM_SC,DF_SC);
                correspondences_sc = match_3DSIFTdescriptors(DM_SC',DF_SC');
                
                fprintf('SIFT-only correspondences get %i matches, SC-only gets %i matches\n',...
                    size(correspondences_sift,2),size(correspondences_sc,2));
                
                correspondences_combine = [correspondences_sc,correspondences_sift]';
                [correspondences,~,~] = unique(correspondences_combine,'rows');
                correspondences = correspondences';
                fprintf('There unique %i matches if we take the union of the two methods\n', size(correspondences,2));
                
            end
            
            if 0
                %correspondences = vl_ubcmatch([DM_SC; DM_SIFT_norm'],[DF_SC; DF_SIFT_norm']);
                correspondences = match_3DSIFTdescriptors([DM_SC; DM_SIFT_norm']',[DF_SC; DF_SIFT_norm']');
            end
            if 1
                correspondences=correspondences_sift;
            end
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
            
            
            % ----------- END ---------- %
        end
    end
    
    save(output_keys_filename,'keyM_total','keyF_total');
else %if we'va already calculated keyM_total and keyF_total, we can just load it
    disp('KeyM_total and KeyF_total already calculated. Skipping');
    
end

end


