% loadParameters;
bigExpParams;

if ~exist(bigparams.REGISTRATION_WORKINGDIR,'dir')
    mkdir(bigparams.REGISTRATION_WORKINGDIR)
end
%% For all fields of view and all rounds, load the keypoints, make global
% Load the reference round

ROUND=bigparams.REFERENCE_ROUND;
filename_fixedkeypoints = fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('registration_allkeys_round%.3i.mat',ROUND));
% Load all the keys
numTiles = prod(bigparams.EXPERIMENT_TILESIZE);

if exist(filename_fixedkeypoints,'file')
    load(filename_fixedkeypoints);
else
    keys_all = cell(numTiles,1);
    for row = 1:size(bigparams.TILE_MAP,1)
        for col = 1:size(bigparams.TILE_MAP,2)
            
            fov_inputnum = bigparams.TILE_MAP(row,col);
            
            %Skipping FOVS we know that are empty by simply looking at it
            if ismember(fov_inputnum,bigparams.FOVS_TO_IGNORE)
                fprintf('Skipping FOV: %i\n',fov_inputnum);
                continue
            end
            
            keys = loadFOVKeyptsAndFeatures(fov_inputnum,ROUND,bigparams);
            fprintf('Adding %i entries from FOV %i\n',length(keys),fov_inputnum);
            keys_all{fov_inputnum+1} = keys;
            
        end
    end
    
    % Generate the fixed keypoints
    %Load the entire fixed reference point
    keys_fixed = {};
    keys_ctr = 1;
    %copy all the keys into one large vector of cells
    %For the fixed/reference, we only work with global, and overwrite the xyz
    %that will be used in the affine calculations
    FOVS = [];
    for f=1:length(keys_all)
        for idx = 1:length(keys_all{f})
            keys_fixed{keys_ctr} = keys_all{f}{idx};
            keys_fixed{keys_ctr}.x = keys_fixed{keys_ctr}.x_global;
            keys_fixed{keys_ctr}.y = keys_fixed{keys_ctr}.y_global;
            keys_fixed{keys_ctr}.z = keys_fixed{keys_ctr}.z_global;
            keys_fixed{keys_ctr} = rmfield(keys_fixed{keys_ctr},'x_global');
            keys_fixed{keys_ctr} = rmfield(keys_fixed{keys_ctr},'y_global');
            keys_fixed{keys_ctr} = rmfield(keys_fixed{keys_ctr},'z_global');
            FOVS(keys_ctr) = keys_fixed{keys_ctr}.F; %Keep track for filtering keypts
            keys_ctr = keys_ctr+ 1;
        end
    end
    
    pos_fixed = zeros(length(keys_fixed),3);
    for k = 1:length(keys_fixed)
        pos_fixed(k,:) = [keys_fixed{k}.x,keys_fixed{k}.y,keys_fixed{k}.z];
    end
    
    fprintf('Starting to save data...'); tic
    save(filename_fixedkeypoints,'keys_all','keys_fixed','FOVS','pos_fixed','-v7.3');
    fprintf('Done!'); toc
    
end

%% Calculating affine transforms from moving_round to neighborhood of fixed
% For each FOV_mov for each round (round_mov)



% Process all the FOVs
for round_mov = 1:7
    if round_mov == bigparams.REFERENCE_ROUND
        continue
    end
    
    %For validation, plot all the fixed points
    %then hold the figure and plot all the moving fovs over the fixed round
    figure('Visible','off')
    plot(pos_fixed(:,2),pos_fixed(:,1),'r.','MarkerSize',15);
    hold on;
    for FOV_mov = 0:numTiles-1
        
        if ismember(FOV_mov,bigparams.FOVS_TO_IGNORE )
            continue;
        end
        
        fovname = sprintf('%s-F%.3i',bigparams.EXPERIMENT_NAME,FOV_mov);
        output_keys_filename = fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('globaltransform_%s_round%03d.mat',fovname,round_mov));
        if exist(output_keys_filename,'file')
            fprintf('Skipping because already processed\n');
            load(output_keys_filename)
            %Plot it anyway
            pos_moving = keyM_total_transformed(:,1:2);
            %NOTE the XY swap!!
            plot(pos_moving(:,1),pos_moving(:,2),'.','MarkerSize',15);
            continue;
        end
        fprintf('Processing FOV%.3i, round %i\n',FOV_mov, round_mov);
        
        keys = loadFOVKeyptsAndFeatures(FOV_mov,round_mov,bigparams);
        
        if isempty(keys)
            continue
        end
        
        
        keys_moving = cell(length(keys),1);
        for idx = 1:length(keys)
            keys_moving{idx} = keys{idx};
            keys_moving{idx}.x = keys_moving{idx}.x_global;
            keys_moving{idx}.y = keys_moving{idx}.y_global;
            keys_moving{idx}.z = keys_moving{idx}.z_global;
        end
        
        %Get the subset of the fixed features that is within one fov around
        %the moving fov
        [row,col] = find(bigparams.TILE_MAP==FOV_mov);
        row_queries = max(1,row-1):min(row+1,size(bigparams.TILE_MAP,1));
        col_queries = max(1,col-1):min(col+1,size(bigparams.TILE_MAP,2));
        fovs_in_nbd = bigparams.TILE_MAP(row_queries,col_queries);
        %Get all the fixed_rnd keys from the local subset
        keys_fixed_subset = keys_fixed(ismember(FOVS,fovs_in_nbd));
        
        %Calculate correspondences between a single moving fov and the
        %neighborhood. (Should take about 30 seconds to find the match)
        try
            [keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed_subset);
        catch
            fprintf("failed to find sufficient correspondences, skip!\n")
            continue
        end
        fprintf('number size of matching keypoints: %i\n',size(keyM_total,1));
        
        %Calculate the affine tform and get back the transformed keypoints
        %This included the RANSAC so the number of correspondences used
        %will be less than the size of keyM_total
        [affine_tform,keyM_total_transformed]  = getGlobalAffineFromCorrespondences(keyM_total,keyF_total);
        
        
        %Plot these warped positions
        pos_moving = keyM_total_transformed(:,1:2);
        %NOTE the XY swap!!
        plot(pos_moving(:,1),pos_moving(:,2),'.','MarkerSize',15);
        
        save(output_keys_filename,'keyM_total','keyF_total','affine_tform','keyM_total_transformed');
    end
    saveas(gcf,fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('all_rounds_warped_round%.3i.fig',round_mov)))
    close
end

%% Registration step 2: finding which moving FOVs match to which fixed FOV
% For each tile, we can use the fixed keypoints to find the original FOV.
% For a minimum number of involved keypoints, we can then later require
% that the FOV will be included when we register
MIN_KEYPOINTS = 10; %We require a minimum set to consider it a good

%Make a big holder for all the moving FOVs implicated for each FOV of the
%fixed round. Note that the indexing has to be +1, but the entries must be
%0-indexed
FOVS_per_fixed_fov_total = cell(numTiles,bigparams.NUMROUNDS);
for round_mov = 2:7
    fprintf('Processing round %i... \n',round_mov);
    for FOV_mov = 0:numTiles-1
        
        if ismember(FOV_mov,bigparams.FOVS_TO_IGNORE )
            continue;
        end
        fovname = sprintf('%s-F%.3i',bigparams.EXPERIMENT_NAME,FOV_mov);
        %Load the results of the neighborhood-based registration
        %'keyM_total','keyF_total','affine_tform','keyM_total_transformed'
        output_keys_filename = fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('globaltransform_%s_round%03d.mat',fovname,round_mov));
        
        if ~exist(output_keys_filename,'file')
            fprintf('No registered result : %s\n',output_keys_filename)
            continue;
        end
        
        load(output_keys_filename);
        
        %Find the fovs that are used in the keyF_total by using a nearest
        %neighbors search to. Note that pos_fixed was plotted (2,1) and
        %pos_moving is plotted (1,2), so we have to toggle one of the positions
        fixed_indices_match_to_movFOV= knnsearch(pos_fixed,keyF_total(:,[2 1 3]));
        
        FOVs_fixed = FOVS(fixed_indices_match_to_movFOV);
        %Count all the FOVs that have the minimum number of matching
        %keypoints. Remember that the number of matching keypoints will
        %likely increase once we're doing an affine tform against just a
        %single FOV: the lack of stitching between fixed FOVs makes it a
        %hard affine alignment
        u=unique(FOVs_fixed);
        [n,~]=histc(FOVs_fixed,u); %Get the counts 
        FOVs_fixed_filtered=u(n>MIN_KEYPOINTS); %filter by counts
        
        %FOVs_fixed_filtered is the list of fixed FOVs that have a
        %significnt amount of overlap with FOV_mov in round_mov. So, we
        %want to make a look up table of all the moving fovs for a given
        %fixed round. So, for each FOVs_fixed_filtered, we add FOV_mov at
        %round_mov
        for fov_f = FOVs_fixed_filtered
            %append the current list with the FOV_mov. 
            %REMEMBER: FOVS_per_fixed_fov_total has to be 1+ bc MATLAB.
            curr_list = FOVS_per_fixed_fov_total{fov_f+1,round_mov};
            curr_list(end+1)= FOV_mov;
            FOVS_per_fixed_fov_total{fov_f+1,round_mov} = curr_list;
        end
        
        %This was the previous mode, which I think was backwards
        %FOVS_per_fixed_fov_total{FOV_mov+1,round_mov} = FOVs_fixed_filtered;
    end
end

% As a post-hoc correction, it's possible the original FOV target (assuming
% no tissue deformation) might have been lost because of the consideration
% of other fixed FOVs that were not stitched. This would mean it impossible
% to get a proper affine registration, and keypoints could be erroneously
% lost. Add back in the fixed FOV number just in case, can always skip it
% if there's no good match

for FOV_fixed = 0:numTiles-1
    for round_mov = 2:7
        fovmatches = FOVS_per_fixed_fov_total{FOV_fixed+1,round_mov};
        if ~ismember(FOV_fixed,fovmatches)
            fovmatches(end+1) = FOV_fixed;
            FOVS_per_fixed_fov_total{FOV_fixed+1,round_mov}=fovmatches;
        end
    end
end


%Tell a story about a single fixed FOV and how it matches to different fovs
%across time
for FOV_mov = 0:numTiles-1
    fprintf('fixed FOV%.3i matches the following:\n',FOV_mov);
    for round_mov = 2:7
        fprintf('\tRound%.3i: %s\n',round_mov,mat2str(FOVS_per_fixed_fov_total{FOV_mov+1,round_mov}));
    end
end


%% To complete this test, we have to then calculate the matches and
%the warps using *only* the moving rounds for a specific fixed FOV

parfor round_mov = 2:7
    global_warpingTEMP(round_mov,keys_fixed,pos_fixed,FOVS,FOVS_per_fixed_fov_total,bigparams)
end

%% Now apply the warps that we've calculated

for FOV_fixed = 0:numTiles-1
    for round_mov = 2:7
        fovmatches = FOVS_per_fixed_fov_total{FOV_fixed+1,round_mov};
        
        
        performAffineTransforms_global(fixed_fov, fovmatches, moving_round,do_downsample,bigparams);
        
    end
end


