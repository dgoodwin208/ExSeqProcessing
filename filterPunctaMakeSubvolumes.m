loadParameters;

params.registeredImagesDir =  '/home/dgoodwin/simulator/simulation_output/';
params.transcriptResultsDir = '/home/dgoodwin/simulator/simulation_output/';
params.punctaSubvolumeDir =   '/home/dgoodwin/simulator/simulation_output/';
params.FILE_BASENAME = 'simseqtryone';
%This loads puncta_centroids and puncta_voxels (the list of voxel INDICES)
%per puncta
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',FILEROOT_NAME_INPUT));
load(filename_centroids)

%%
DISTANCE_THRESHOLD = 10;
REF_IDX = 5;

%load the data for the reference round
puncta_ref = puncta_centroids{REF_IDX}; 
% interpolate the Z position so we calculate the nearest neighbor
% isotropically
puncta_ref(:,3) = puncta_ref(:,3)*(.5/.165);

%Puncta matches is the same size as a transcript for a specific puncta,
%indexed from the puncta in the reference round, but in this case each
%round entry is the index of the puncta object from that round that is the
%nearest neighbor match
puncta_matches = zeros(size(puncta_ref,1),params.NUM_ROUNDS);
%same thing but track distances to the puncta across rounds
puncta_matches_distances = zeros(size(puncta_ref,1),params.NUM_ROUNDS);

for mov_idx = 1:params.NUM_ROUNDS
    
    %No need to run nearest neighbor to itself. Just set the index to
    %itself. Somewhat redundant but will make code easier later
    if mov_idx==REF_IDX
        puncta_matches(:,mov_idx) = 1:size(puncta_matches);
        continue;
    end
    
    
    puncta_mov = puncta_centroids{mov_idx}; 
    puncta_mov(:,3) = puncta_mov(:,3)*(.5/.165);
    
    %init the holder for this round
    punctamaps = {};
        
    %returns an my-by-1 vector D containing the distances between each
    %observation in Y and the corresponding closest observation in X.
    %That is, D(i) is the distance between X(IDX(i),:) and Y(i,:)
    % [IDX,D] = knnsearch(X,Y,'K',3);
    [IDX,D] = knnsearch(puncta_ref,puncta_mov,'K',1); %getting two other neighbor options
    %So puncta_mov(IDX(i),:) -> puncta_ref(i,:)
    %Note that IDX(i) can be a non-unique value
    
    %Right now we're doing the most naive way, allowing a puncta from the
    %reference round to potentially hit the same moving-round puncta. fine.
    output_ctr = 1;
    num_discarded_noneighbors = 0;
    num_discarded_distance = 0;
    
    %confusing but ref_idx is the puncta index in the reference round
    for ref_idx = 1:size(puncta_ref,1)
        indices_of_MOV = find(IDX == ref_idx);
        
        if isempty(indices_of_MOV)
            %fprintf('Skipping due to no matches to ref_idx=%i \n',ref_idx);
            num_discarded_noneighbors = num_discarded_noneighbors+1;
            continue
        end
        
        distances_to_REF = D(indices_of_MOV);
        
        puncta_matches(ref_idx,mov_idx) = 0;
        
        if length(indices_of_MOV) == 1
            puncta_matches(ref_idx,mov_idx) = indices_of_MOV(1);
            puncta_matches_distances(ref_idx,mov_idx) = distances_to_REF(1);
%             punctamap.pos_moving = puncta_mov(indices_of_MOV(1),:);
%             punctamap.index_mov = indices_of_MOV(1);
%             punctamap.distance = distances_to_REF(1);
%             punctamap.neighbors = 1;
        else
            [distances_sorted,I] = sort(distances_to_REF,'ascend');
%             punctamap.pos_moving = puncta_mov(indices_of_MOV(I(1)),:);
%             punctamap.distance = distances_to_REF(1);
%             punctamap.index_mov = indices_of_MOV(I(1));
%             punctamap.neighbors = numel(distances_to_REF);
            puncta_matches(ref_idx,mov_idx) = indices_of_MOV(I(1));
            puncta_matches_distances(ref_idx,mov_idx) = distances_sorted(1);

        end
        
        if puncta_matches_distances(ref_idx,mov_idx)>DISTANCE_THRESHOLD
            puncta_matches_distances(ref_idx,mov_idx) = 0;
            num_discarded_distance= num_discarded_distance+1;
        end
        
%         if mod(ref_idx,1000)==0
%             fprintf('Looped %i/%i puncta for rnd=%i\n',ref_idx,size(puncta_ref,1),mov_idx);
%         end
    end
    fprintf('Discarded %i no-neighbor and %i remote puncta in rnd=%i\n',num_discarded_noneighbors,num_discarded_distance,mov_idx);
end
%%

MIN_NEIGHBOR_AGREEMENT = 18;

puncta_indices_filtered = (sum(puncta_matches~=0,2)>=MIN_NEIGHBOR_AGREEMENT);
num_puncta_filtered = sum(puncta_indices_filtered);

%As a proof of concept, make an image that shows those puncta
filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,1,'ch00'));
sample_img = load3DTif_uint16(filename_in);

data_height = size(sample_img,1);
data_width = size(sample_img,2);
data_depth = size(sample_img,3);

filtered_mask = zeros(size(sample_img));

for ref_idx = 1:size(puncta_matches,1)
    %skip any ones that are filtered out
    if ~puncta_indices_filtered(ref_idx); continue;end;
    
    pixel_list = puncta_voxels{ref_idx};
    
    filtered_mask(pixel_list) = 200;
end

save3DTif_uint16(filtered_mask,fullfile(params.registeredImagesDir,sprintf('%s_filtered_punctarois.tif',params.FILE_BASENAME)));

%% Make the 10x10x10 subvolumes we started this with, but now only with the pixels from the puncta!

chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS,1);

parfor exp_idx = 1:params.NUM_ROUNDS
    disp(['round=',num2str(exp_idx)])
    
    %Load all channels of data into memory for one experiment
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    
    
    for c_idx = params.COLOR_VEC
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,exp_idx,chan_strs{c_idx}));
        experiment_set(:,:,:,c_idx) = load3DTif_uint16(filename_in);         
    end
    
    disp(['[',num2str(exp_idx),'] processing puncta in parallel'])
    
    puncta_set_cell{exp_idx} = cell(params.NUM_CHANNELS,num_puncta_filtered);
    
    %the puncta indices are here in linear form for a specific round
    punctafeinder_indices = puncta_voxels{exp_idx};
    experiment_set_masked = zeros(data_height,data_width,data_depth);
    
    puncta_mov = puncta_centroids{exp_idx}; 
    
    subvolume_ctr = 1;
    
    %Note that this puncta is the REFERENCE round puncta
    %So we use the puncta_matches array to get the moving index
    for puncta_idx = 1:size(puncta_matches,1)
        
        %Skip the puncta that failed the filtering
        if ~puncta_indices_filtered(puncta_idx); continue;end;
        
        %Get the puncta_idx in the context of this experimental round
        moving_puncta_idx = puncta_matches(puncta_idx,exp_idx);
        
        %If this round did not have a match for this puncta, just return
        %all zeros 
        if moving_puncta_idx==0
            puncta_set_cell{exp_idx}{c_idx,subvolume_ctr} = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE);
            subvolume_ctr = subvolume_ctr+1;
            continue;
        end
        
        %Get the centroid's Y X Z
        Y = round(puncta_mov(moving_puncta_idx,1)); 
        X = round(puncta_mov(moving_puncta_idx,2));
        Z = round(puncta_mov(moving_puncta_idx,3));
        %Because of interpolation in the calculation of puncta location, we
        %have to undo the interpoaltino here:
        Z = round(Z * (.165/.5));
        
        %If we were just drawing a 10x10x10 subregion around the
        %puncta, we'd do this
        y_indices = Y - params.PUNCTA_SIZE/2 + 1: Y + params.PUNCTA_SIZE/2;
        x_indices = X - params.PUNCTA_SIZE/2 + 1: X + params.PUNCTA_SIZE/2;
        z_indices = Z - params.PUNCTA_SIZE/2 + 1: Z + params.PUNCTA_SIZE/2;
        
        
        
        
        %These are the indices for the puncta in question
        punctafeinder_indices_for_puncta = punctafeinder_indices{moving_puncta_idx};
        %now converted to 3D
        [i1, i2, i3] = ind2sub(size(experiment_set_masked),punctafeinder_indices_for_puncta); 
        
        for c_idx = params.COLOR_VEC
            %This is admittedly heavy handed, but it's a quick prototype
            %esp for the cropped dataset:
%             experiment_set_channel = experiment_set(:,:,:,c_idx);
            
            %This makes a volume that is all zeros except for 
            experiment_set_masked(i1,i2,i3) = experiment_set(i1,i2,i3,c_idx);
            
            %So we can AND these indices with the PixelIdxList from the
            %centroids, then use those locations to capture the puncta
            try
                puncta_set_cell{exp_idx}{c_idx,subvolume_ctr} = experiment_set_masked(y_indices,x_indices,z_indices);
            catch
                %In the case of indices outside the volume, just return
                %zeros for now
                puncta_set_cell{exp_idx}{c_idx,subvolume_ctr} = ...
                    zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE); 
            end
            
            %Reset the specific pixels of the mask
            experiment_set_masked(i1,i2,i3) = 0;
        end
        subvolume_ctr = subvolume_ctr+1;
        
        if mod(subvolume_ctr,100)==0
           fprintf('Rnd %i, Puncta %i processed\n',exp_idx,subvolume_ctr); 
        end
    end
end

disp('reducing processed puncta')
% reduction of parfor
for exp_idx = 1:params.NUM_ROUNDS
    for puncta_idx = 1:num_puncta_filtered
        for c_idx = params.COLOR_VEC
            if ~isequal(size(puncta_set_cell{exp_idx}{c_idx,puncta_idx}), [0 0])
                puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,puncta_idx};
            end
        end
    end
end


disp('saving files from makePunctaVolumes')
%just save puncta_set
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)),...
    'puncta_set','-v7.3');

%%








% save(fullfile(params.punctaSubvolumeDir,sprintf('%s_centroid_collections.mat',params.FILE_BASENAME)),'centroid_collections','-v7.3');
% save(fullfile(params.punctaSubvolumeDir,sprintf('%s_punctamap.mat',params.FILE_BASENAME)),'punctamap_master','-v7.3');

