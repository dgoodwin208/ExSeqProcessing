loadParameters;

%params.FILE_BASENAME = 'simseqtryone';
%This loads puncta_centroids and puncta_voxels (the list of voxel INDICES)
%per puncta
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids)

%%
DISTANCE_THRESHOLD = 10;
REF_IDX = 5;

%load all the puncta locations for the reference round
puncta_ref = puncta_centroids{REF_IDX};

% interpolate the Z position so we calculate the nearest neighbor
% isotropically
%puncta_ref(:,3) = puncta_ref(:,3)*(.5/.165);

%Puncta matches is the same size as a transcript for a specific puncta,
%indexed from the puncta in the reference round, but in this case each
%round entry is the index of the puncta object from that round that is the
%nearest neighbor match
puncta_matches = zeros(size(puncta_ref,1),params.NUM_ROUNDS);
%same thing but track distances to the puncta across rounds
puncta_matches_distances = zeros(size(puncta_ref,1),params.NUM_ROUNDS);

for mov_rnd_idx = 1:params.NUM_ROUNDS
    
    %No need to run nearest neighbor to itself. Just set the index to
    %itself. Somewhat redundant but will make code easier later
    if mov_rnd_idx==REF_IDX
        puncta_matches(:,mov_rnd_idx) = 1:size(puncta_matches);
        continue;
    end
    
    
    puncta_mov = puncta_centroids{mov_rnd_idx};
    %puncta_mov(:,3) = puncta_mov(:,3)*(.5/.165);
    
    %init the holder for this round
    punctamaps = {};
    
    %returns an my-by-1 vector D containing the distances between each
    %observation in Y and the corresponding closest observation in X.
    %That is, D(i) is the distance between X(IDX(i),:) and Y(i,:)
    % [IDX,D] = knnsearch(X,Y,'K',3);
    [IDX,D] = knnsearch(puncta_ref,puncta_mov,'K',5); %getting five other options
    
    %create the distance matrix that is populated by IDX and D
    A = sparse(size(puncta_mov,1),size(puncta_ref,1));
    for idx_row = 1:size(IDX,1)
        for idx_col = 1:size(IDX,2)
            %For that row in the IDX, loop over the columns, which are the
            %indices to the reference puncta round
            %The entries are the inverse of distance, which is useful
            %because we're going to get the maximum weighted partition
            A(idx_row,IDX(idx_row,idx_col)) = 1/D(idx_row,idx_col);
        end
    end
    
    %Using a bipartite complete matching algorithm
    [~, matched_indices_moving,matched_indices_ref] = bipartite_matching(A);

    num_discarded_noneighbors = 0;
    num_discarded_distance = 0;
    
    %confusing but ref_idx is the puncta index in the reference round
    for matched_row_idx = 1:length(matched_indices_moving)
        
        %Get the indices of the puncta that were matched using the
        %bipartite graph match
        matched_puncta_mov_idx = matched_indices_moving(matched_row_idx);
        ref_puncta_idx = matched_indices_ref(matched_row_idx);
        
        
        puncta_matches(ref_puncta_idx,mov_rnd_idx) = matched_puncta_mov_idx;
        %Going back to the A matrix (which is indexed like the transpose)
        %to get the original distance value out (has to be re-inverted)
        puncta_matches_distances(ref_puncta_idx,mov_rnd_idx) = 1/A(matched_puncta_mov_idx,ref_puncta_idx);        
        
        %What if the nearest match is too far away?
        if puncta_matches_distances(ref_puncta_idx,mov_rnd_idx)>DISTANCE_THRESHOLD
            puncta_matches_distances(ref_puncta_idx,mov_rnd_idx) = 0;
            num_discarded_distance= num_discarded_distance+1;
        end
    end
    
   
    fprintf('Discarded %i no-neighbor and %i remote puncta in rnd=%i\n',num_discarded_noneighbors,num_discarded_distance,mov_rnd_idx);
end
%%

MIN_NEIGHBOR_AGREEMENT = params.PUNCTA_PRESENT_THRESHOLD;

%Boolean vector that indicates if a puncta index from reference round has
%sufficient amount of matches across other rounds
puncta_indices_filtered = (sum(puncta_matches~=0,2)>=MIN_NEIGHBOR_AGREEMENT);
num_puncta_filtered = sum(puncta_indices_filtered);

%As a proof of concept, make an image that shows those puncta
filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,1,'ch00'));
sample_img = load3DTif_uint16(filename_in);

data_height = size(sample_img,1);
data_width = size(sample_img,2);
data_depth = size(sample_img,3);

filtered_mask = zeros(size(sample_img));

puncta_voxels_refround = puncta_voxels{REF_IDX};
for ref_idx = 1:size(puncta_matches,1)
    %skip any ones that are filtered out
    if ~puncta_indices_filtered(ref_idx); continue;end;
    
    pixel_list = puncta_voxels_refround{ref_idx};
    
    filtered_mask(pixel_list) = 200;
end

save3DTif_uint16(filtered_mask,fullfile(params.punctaSubvolumeDir,sprintf('%s_filtered_punctarois.tif',params.FILE_BASENAME)));

%% Make the 10x10x10 subvolumes we started this with, but now only with the pixels from the puncta!

chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS,1);
pos_cell = cell(params.NUM_ROUNDS,1);
%pos(:,exp_idx,puncta_idx)

parfor exp_idx = 1:params.NUM_ROUNDS
    disp(['round=',num2str(exp_idx)])
    
    %Load all channels of data into memory for one experiment
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    
    
    for c_idx = params.COLOR_VEC
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,exp_idx,chan_strs{c_idx}));
        experiment_set(:,:,:,c_idx) = load3DTif_uint16(filename_in);
    end
    
    disp(['[',num2str(exp_idx),'] processing puncta in parallel'])
    
    puncta_set_cell{exp_idx} = cell(params.NUM_CHANNELS,num_puncta_filtered);
    pos_per_round = zeros(3,num_puncta_filtered);
    %the puncta indices are here in linear form for a specific round
    punctafeinder_indices = puncta_voxels{exp_idx};
    
    
    puncta_mov = puncta_centroids{exp_idx};
    
    subvolume_ctr = 1;
    padwidth = ceil(params.PUNCTA_SIZE/2);
    experiment_set_padded = padarray(experiment_set,[padwidth padwidth padwidth 0],0);
    experiment_set_padded_masked = zeros(size(experiment_set_padded));
    
    %Note that this puncta is the REFERENCE round puncta
    %So we use the puncta_matches array to get the moving index
    %Puncta_idx here is the index of the puncta in the reference round
    for puncta_idx = 1:size(puncta_matches,1)
        
        %Skip the puncta that failed the filtering
        if ~puncta_indices_filtered(puncta_idx); continue;end;
        
        %Get the puncta_idx in the context of this experimental round
        moving_puncta_idx = puncta_matches(puncta_idx,exp_idx);
        
        %If this round did not have a match for this puncta, just return
        %all zeros
        if moving_puncta_idx==0
            for c_idx = params.COLOR_VEC
                puncta_set_cell{exp_idx}{c_idx,subvolume_ctr} = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE)-1;
            end
            subvolume_ctr = subvolume_ctr+1;
            fprintf('No matching puncta for puncta_idx=%i in round %i\n',puncta_idx,exp_idx)
            continue;
        end
        
        %Get the centroid's Y X Z
        %NOTE: The centroid position come from the regionprops() call in
        %punctafeinder.m and have the XY coords flipped relative to what
        %we're used to, so Y and X are switched
        Y = round(puncta_mov(moving_puncta_idx,2))+padwidth; 
        X = round(puncta_mov(moving_puncta_idx,1))+padwidth;
        Z = round(puncta_mov(moving_puncta_idx,3))+padwidth;
        %Because of interpolation in the calculation of puncta location, we
        %have to undo the interpoaltino here:
        %Z = round(Z * (.165/.5));
        
        %If we were just drawing a 10x10x10 subregion around the
        %puncta, we'd do this
        y_indices = Y - params.PUNCTA_SIZE/2 + 1: Y + params.PUNCTA_SIZE/2;
        x_indices = X - params.PUNCTA_SIZE/2 + 1: X + params.PUNCTA_SIZE/2;
        z_indices = Z - params.PUNCTA_SIZE/2 + 1: Z + params.PUNCTA_SIZE/2;

                
        %These are the indices for the puncta in question
        punctafeinder_indices_for_puncta = punctafeinder_indices{moving_puncta_idx};
        %now converted to 3D in the original coordinate space 
        [i1, i2, i3] = ind2sub(size(experiment_set),punctafeinder_indices_for_puncta);
        %shift the conversions with the padwidth
        i1 = i1+padwidth; i2 = i2+padwidth; i3 = i3+padwidth;
        
        for c_idx = params.COLOR_VEC
            %This makes a volume that is all zeros except for the punctafeinder_indices_for_puncta
            experiment_set_padded_masked(i1,i2,i3) = experiment_set_padded(i1,i2,i3,c_idx);
            
            pixels_for_puncta_set = experiment_set_padded_masked(y_indices,x_indices,z_indices);
            
            if max(pixels_for_puncta_set(:))==0
               fprintf('Ok we have found an issue\n');
               barf()
            end
            %Then we take the PUNCTA_SIZE region around those pixels only
            puncta_set_cell{exp_idx}{c_idx,subvolume_ctr} = pixels_for_puncta_set;
            
            %Reset the specific pixels of the mask
            experiment_set_padded_masked(i1,i2,i3) = 0;
        end
        
        
       pos_per_round(:,subvolume_ctr) = [Y X Z]-padwidth;
        
        subvolume_ctr = subvolume_ctr+1;
        if mod(subvolume_ctr,400)==0
            fprintf('Rnd %i, Puncta %i processed\n',exp_idx,subvolume_ctr);
        end
    end
    pos_cell{exp_idx} =pos_per_round;
end

disp('reducing processed puncta')
puncta_set = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.NUM_ROUNDS,params.NUM_CHANNELS,num_puncta_filtered);

% reduction of parfor
for exp_idx = 1:params.NUM_ROUNDS
    for puncta_idx = 1:num_puncta_filtered
        for c_idx = params.COLOR_VEC            
            puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,puncta_idx};
        end
    end
end
pos= zeros(3,params.NUM_ROUNDS,num_puncta_filtered);
for exp_idx = 1:params.NUM_ROUNDS
    pos_per_round = pos_cell{exp_idx};
    for puncta_idx = 1:num_puncta_filtered
        pos(:,exp_idx,puncta_idx) = pos_per_round(:,puncta_idx);
    end
end


disp('saving files from makePunctaVolumes')
%just save puncta_set
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)),...
    'puncta_set','pos','-v7.3');

%% Checking the quality of the resulting puncta_set

empty_puncta_ctr = 0;
for p_idx = 1:size(puncta_set,6)
   puncta = puncta_set(:,:,:,:,:,p_idx);
%    fprintf('idx: %i, maxval: %i\n',p_idx,max(puncta(:)));
   if max(puncta(:))==0
       empty_puncta_ctr = empty_puncta_ctr+1;
   end
end


empty_puncta_ctr

%% Make plots of distances to puncta
figure
histogram(puncta_matches_distances(:,4),20)
title('Histogram of distances of ref=groundTruth and mov=sim round 5');
xlabel('Distances in pixels')
ylabel('Count');



