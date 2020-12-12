% loadParameters;
bigExpParams;

%% For all fields of view and all rounds, load the keypoints, make global
% Load the reference round


% Load all the keys

for ROUND = 1:2 %:bigparams.NUMROUNDS
    
    keys_all = cell(numTiles,1);
    for row = 1:size(tileMap_indices_reference,1)
        for col = 1:size(tileMap_indices_reference,2)
            
            fov_inputnum = tileMap_indices_reference(row,col);
            
            %Skipping FOVS we know that are empty by simply looking at it
            if ismember(fov_inputnum,bigparams.FOVS_TO_IGNORE)
                fprintf('Skipping FOV: %i\n',fov_inputnum);
                continue
            end
                
            
            keys_all{fov_inputnum+1} = loadFOVKeyptsAndFeatures(fov_inputnum,ROUND,bigparams)
            
%             foldername = fullfile(bigparams.EXPERIMENT_FOLDERROOT,...
%                 sprintf('F%.3i',fov_inputnum),...
%                 '4_registration',...
%                 sprintf('%s-F%.3i-downsample_round%.3i_%s',...
%                 bigparams.EXPERIMENT_NAME,fov_inputnum,ROUND,bigparams.REG_CHANNEL) );
%             
%             if exist(foldername,'dir')
%                 %Get the file from inside the folder
%                 files = dir(fullfile(foldername,'*.mat'));
%                 if length(files)==0
%                     fprintf('%s is empty\n',foldername);
%                     continue
%                 end
%                 filename = files(1).name;
%                 %Load the variable 'keys'
%                 load(fullfile(foldername,filename))
% 
%                 %Copy those keys into a global holder of all
%                 %keypoints+descriptors
%                 fprintf('Adding %i entries from FOV %i\n',length(keys),fov_inputnum);
%                 for k = 1:length(keys)
%                     
%                     %The position of the keypoints is in
%                     %downsampled coordinatees
%                     
%                     keys{k}.x_global = bigparams.IMG_SIZE_XY(1)*(col-1) + keys{k}.x*bigparams.DOWNSAMPLE_RATE;
%                     keys{k}.y_global = bigparams.IMG_SIZE_XY(2)*(row-1) + keys{k}.y*bigparams.DOWNSAMPLE_RATE;
%                     keys{k}.z_global = keys{k}.z*bigparams.DOWNSAMPLE_RATE;
%                     keys{k}.F = fov_inputnum;
%                     
%                     keys{k} = rmfield(keys{k},'xyScale');
%                     keys{k} = rmfield(keys{k},'tScale');
%                     keys{k} = rmfield(keys{k},'k');
%                 end
%                 keys_all{fov_inputnum+1} = keys;
% 
%             else
%                 fprintf('%s does not exist\n',foldername);
%             end

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
for idx=1:length(data.keys)
        %copy all the keys into one large vector of cells
        keys_fixed{keys_ctr} = keys{idx};
        keys_fixed{keys_ctr}.x = keys{idx}.x;
        keys_fixed{keys_ctr}.y = keys{idx}.y;
        keys_ctr = keys_ctr+ 1;
end

%% Load an "easy" field of view - 40 (60% volume overlap)

%% Load an "hard" field of view - 50 (24% volume overlap)
%% Now load a moving round

%% Try an alignment

[keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed)
