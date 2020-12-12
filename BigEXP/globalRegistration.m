% loadParameters;
bigExpParams;

%% For all fields of view and all rounds, load the keypoints, make global
% Load the reference round


% Load all the keys

for ROUND = 1:1 %:bigparams.NUMROUNDS
    
    keys_all = cell(numTiles,1);
    for row = 1:size(tileMap_indices_reference,1)
        for col = 1:size(tileMap_indices_reference,2)
            
            fov_inputnum = tileMap_indices_reference(row,col);
            
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
    save(fullfile(bigparams.OUTPUTDIR,sprintf('registration_allkeys_round%.3i.mat',ROUND)),'keys_all','-v7.3');
    fprintf('Done!'); toc
    
end % End round

%% Generate the fixed keypoints
%Load the entire fixed reference point
myDIR = '/Users/goody/Neuro/ExSeq/HTAPP_514';
load(fullfile(myDIR, 'registration_allkeys_round001.mat'));
keys_fixed = {};
keys_ctr = 1;
%copy all the keys into one large vector of cells
%For the fixed/reference, we only work with global, and overwrite the xyz
%that will be used in the affine calculations
for f=1:length(keys_all)
    for idx = 1:length(keys_all{f})     
        keys_fixed{keys_ctr} = keys_all{f}{idx};
        keys_fixed{keys_ctr}.x = keys_fixed{keys_ctr}.x_global;
        keys_fixed{keys_ctr}.y = keys_fixed{keys_ctr}.y_global;
        keys_fixed{keys_ctr}.z = keys_fixed{keys_ctr}.z_global;
        keys_fixed{keys_ctr} = rmfield(keys_fixed{keys_ctr},'x_global');
        keys_fixed{keys_ctr} = rmfield(keys_fixed{keys_ctr},'y_global');
        keys_fixed{keys_ctr} = rmfield(keys_fixed{keys_ctr},'z_global');
        keys_ctr = keys_ctr+ 1;
    end
end

%% Load an "easy" field of view - 40 (60% volume overlap)

keys = loadFOVKeyptsAndFeatures(40,3,bigparams);
%As a quick development hack, just scale up the original xyz coords
keys_moving = {};
keys_ctr = 1;

for idx = 1:length(keys)     
    keys_moving{keys_ctr} = keys{idx};
    keys_moving{keys_ctr}.x = keys_fixed{keys_ctr}.x*bigparams.DOWNSAMPLE_RATE;
    keys_moving{keys_ctr}.y = keys_fixed{keys_ctr}.y*bigparams.DOWNSAMPLE_RATE;
    keys_moving{keys_ctr}.z = keys_fixed{keys_ctr}.z*bigparams.DOWNSAMPLE_RATE;

    keys_ctr = keys_ctr+ 1;
end



%% Load an "hard" field of view - 50 (24% volume overlap)
%% Now load a moving round

%% Try an alignment

[keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed)
