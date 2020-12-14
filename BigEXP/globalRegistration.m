% loadParameters;
bigExpParams;

%% For all fields of view and all rounds, load the keypoints, make global
% Load the reference round

ROUND=bigparams.REFERENCE_ROUND;
filename_fixedkeypoints = fullfile(bigparams.OUTPUTDIR,sprintf('registration_allkeys_round%.3i.mat',ROUND));
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
    
    fprintf('Starting to save data...'); tic
    save(filename_fixedkeypoints,'keys_all','-v7.3');
    fprintf('Done!'); toc

end
%% Generate the fixed keypoints
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

%% Load an "easy" field of view - 40 (60% volume overlap)
round_mov = 3;


FOV_mov = 50;
keys = loadFOVKeyptsAndFeatures(FOV_mov,round_mov,bigparams);
%As a quick development hack, just scale up the original xyz coords

keys_moving = cell(length(keys),1);
for idx = 1:length(keys)
    keys_moving{idx} = keys{idx};
    keys_moving{idx}.x = keys_moving{keys_ctr}.x*bigparams.DOWNSAMPLE_RATE;
    keys_moving{idx}.y = keys_moving{keys_ctr}.y*bigparams.DOWNSAMPLE_RATE;
    keys_moving{idx}.z = keys_moving{keys_ctr}.z*bigparams.DOWNSAMPLE_RATE;
end

%Get the subset of the fixed features
[row,col] = find(bigparams.TILE_MAP==FOV_mov);
row_queries = max(1,row-1):min(row+1,size(bigparams.TILE_MAP,1));
col_queries = max(1,col-1):min(col+1,size(bigparams.TILE_MAP,2));
fovs_in_nbd = bigparams.TILE_MAP(row_queries,col_queries);

keys_fixed_subset = keys_fixed(ismember(FOVS,fovs_in_nbd));

%Should take about 30 seconds to find the match
[keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed_subset);

affine_tform  = getGlobalAffineFromCorrespondences(

fovname = sprintf('%s-F%.3i',bigparams.EXPERIMENT_NAME,FOV_mov);
output_keys_filename = fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('globalkeys_%s_round%03d.mat',fovname,FOV_mov));
save(output_keys_filename,'keyM_total','keyF_total');

%