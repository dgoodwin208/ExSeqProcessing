% This version of puncta extraction uses the solution written by Zachary
% Chiang in the Payne, Reginato and Chiang 2020 In Situ Genome Sequencing
% The original code can be accessed here: 
% https://github.com/zchiang/in_situ_genome_sequencing_processing

loadParameters;
% The core idea is instead of amplifying punctate signals and then
% thresholding and watershedding, we instead look for intensity peaks and
% then threshold on correlated nearby pixels
%Define parameters up top
ROUNDS = 1:params.NUM_ROUNDS;
%Divide the sample up into a few sections
SPLIT_IMG_MAXIMA = 2; %Finding maxima requires less memory than spot calling
SPLIT_IMG_CORR = 2;

%Note: For the 2048x2048x200 size images we used in the Boyden Lab, each
%core can take 60GB when we used SPLIT_IMG_MAXIMA=SPLIT_IMG_CORR=2 
parpool(max(SPLIT_IMG_CORR,SPLIT_IMG_MAXIMA));

%We're adding the normalized channels, and for our microscope, the pixel
%offset is 100.
noise_floor = params.NUM_CHANNELS*100*params.NUM_ROUNDS;
thresh = 1.25*noise_floor;
%Set an upper bound of how many puncta we can extract by determining the
%average size of amplicons discovered in the data. Using the maximum size
%of puncta (set at 2000voxels at this writing), we want to make sure that
%we are not getting so many possible reads that we're getting less than
%2000px average vol size
MINAVERAGE_PUNCTAVOL = params.PUNCTA_SIZE_MAX;

%These are the parameters taken directly from Zachary Chiang's code, which
%worked well straight out of the box.
corr_thresh = 0.95;
spot_dim = [5 5 5]; %This creates an 11x11x11 window around the maxima

%Loop over all rounds to sum up the SummedNorm channels
for roundnum = ROUNDS
    summed_norm = load3DImage_uint16(fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_%s.%s',params.FILE_BASENAME,roundnum,params.PUNCTA_CHANNEL_SEG,regparams.REGISTRATION_TYPE,params.IMAGE_EXT)));
    summed_norm(summed_norm(:)>prctile(summed_norm(:),99.9)) = prctile(summed_norm(:),99.9);
    if ~exist('total_summed_norm','var')
        total_summed_norm = summed_norm;
        
        %The zero_mask_tracker will count 0 values, which is the
        %indicator of being outside of the registered volume
        zero_mask_tracker = zeros(size(total_summed_norm));
        zero_mask_tracker = zero_mask_tracker + (total_summed_norm==0);
    else
        total_summed_norm = total_summed_norm + summed_norm;
        zero_mask_tracker = zero_mask_tracker + (summed_norm==0);
    end
    
end

% Convert the zero_mask into a cropping bounds
mask = zero_mask_tracker<=params.MAXNUM_MISSINGROUND;
%1 = there was less than params.MAXNUM_MISSINGROUND of 0s
crop_dims = zeros(3,2); %num_dims x min/max
for dim = 1:3
    %Get the dimensions to take the maximums of
    dims_to_mip = 1:3;
    dims_to_mip(dim) = [];
    %Do the max twice
    max_mip = max(mask,[],dims_to_mip(1));
    max_mip = max(max_mip ,[],dims_to_mip(2));
    %Max_mip should now be a vector
    %We can get the start and end of the acceptable range
    crop_dims(dim,1) = find(max_mip ,1,'first');
    crop_dims(dim,2) = find(max_mip ,1,'last');
end

%Now crop the data that we will be applying the DOG to
img = total_summed_norm(...
    crop_dims(1,1):crop_dims(1,2),...
    crop_dims(2,1):crop_dims(2,2),...
    crop_dims(3,1):crop_dims(3,2));

%Note how big the image size is after cropping
imgdim = size(img);

y_indices = floor(linspace(1,imgdim(1),SPLIT_IMG_MAXIMA+1));
x_indices = floor(linspace(1,imgdim(2),SPLIT_IMG_MAXIMA+1));


%Then for each section, get all the maxima from the combined image of
%summedNorm

%First, we use a cell array to split up the images for parallellization
cell_subimg = cell(SPLIT_IMG_MAXIMA,SPLIT_IMG_MAXIMA);
cell_offsets = cell(SPLIT_IMG_MAXIMA,SPLIT_IMG_MAXIMA);
for y_idx = 1:SPLIT_IMG_MAXIMA
    y_range = y_indices(y_idx):y_indices(y_idx+1);
    for x_idx= 1:SPLIT_IMG_MAXIMA
        x_range = x_indices(x_idx):x_indices(x_idx+1);
        
        offsets = [y_range(1)-1,x_range(1)-1,0];
        subimg = img(y_range,x_range,:);
        cell_subimg{y_idx,x_idx} = subimg;
        cell_offsets{y_idx,x_idx} = offsets;
    end
end


%Then we parallelize the maxima finding 
cell_total_maxima = cell(SPLIT_IMG_MAXIMA,SPLIT_IMG_MAXIMA);
parfor y_idx = 1:SPLIT_IMG_MAXIMA
    for x_idx= 1:SPLIT_IMG_MAXIMA
    
        %Get the spatial offsets
        offsets = cell_offsets{y_idx,x_idx};
        %Load the image from the cell array
        subimg = cell_subimg{y_idx,x_idx};
        tic
        %These parameters 1,0 were also taken from Zach Chiang's code cited
        %above
        [Maxima,MaxPos,Minima,MinPos]=MinimaMaxima3D(subimg,1,0);
        elapsed = toc;
        %Only keep the peaks that are above a parameterized value
        %However, due to the variability of the data, we have to
        %additionally check if we are getting too many putative amplicons.
        %In the If clause below, if we get too many puncta, re-adjust the
        %threshold to give us only the N brightest, chosen to give each
        %puncta the desired average volume. 
        if numel(subimg)/sum(Maxima>thresh) < MINAVERAGE_PUNCTAVOL
            numPuncta = floor(numel(subimg)/MINAVERAGE_PUNCTAVOL);
            sortedMaxima = sort(Maxima,'descend');
            newthresh = sortedMaxima(numPuncta); %Get the Nth brightness 
            pos = MaxPos(Maxima>newthresh,:);
        else 
            pos = MaxPos(Maxima>thresh,:);
        end
        cell_total_maxima{y_idx,x_idx} = pos+ offsets;
        
        fprintf('(%i, %i): Total number of peaks for subsection %i. Time: %f \n',...
            y_idx, x_idx, size(pos,1),elapsed);
    end
end

%Consolidate all the puncta into one vector
total_maxima = [];
for y_idx = 1:SPLIT_IMG_MAXIMA
    for x_idx= 1:SPLIT_IMG_MAXIMA
        N = size(cell_total_maxima{y_idx,x_idx},1);
        total_maxima(end+1:end+N,:) = cell_total_maxima{y_idx,x_idx};
    end
end

clear cell_subimg;


%Create a parfor loop to go over the rows of subsection, then we will
%consolidate the cell array at the end.
cell_all_intensities = cell(SPLIT_IMG_CORR,1);
cell_all_sizes = cell(SPLIT_IMG_CORR,1);
cell_all_positions = cell(SPLIT_IMG_CORR,1);
cell_all_voxels = cell(SPLIT_IMG_CORR,1);
for y_idx = 1:SPLIT_IMG_CORR

    %Over-allocate the matrix for the voxel data for speed
    all_intensities = zeros(size(total_maxima,1),params.NUM_ROUNDS,params.NUM_CHANNELS);
    all_sizes = zeros(size(total_maxima,1),1);
    all_positions = zeros(size(total_maxima,1),3);
    all_voxels = cell(size(total_maxima,1),1);
    all_ctr = 1;

    %Divide the sample up into a few sections
    y_indices = floor(linspace(1,imgdim(1),SPLIT_IMG_MAXIMA+1));
    x_indices = floor(linspace(1,imgdim(2),SPLIT_IMG_MAXIMA+1));


    %What is the pixel range of this subsection
    y_range = y_indices(y_idx):y_indices(y_idx+1);
    
    %Loop over the columns in this row
    for x_idx= 1:SPLIT_IMG_CORR
        %What is the pixel range of this subection?
        x_range = x_indices(x_idx):x_indices(x_idx+1);
        offsets = [y_range(1)-1,x_range(1)-1,0];
        
        %Get all the maxima in this region
        maxima_mask =   total_maxima(:,1) > offsets(1) & ...
            total_maxima(:,2) > offsets(2)   & ...
            total_maxima(:,1) <= y_range(end)& ...
            total_maxima(:,2) <= x_range(end);
        fprintf('Starting row %i/%i and col %i/%i\n',y_idx,SPLIT_IMG_MAXIMA,...
            x_idx,SPLIT_IMG_MAXIMA);
        
        %Load all the rounds and all the color channels for this specific
        %data
        all_seq_data = zeros(length(y_range),length(x_range),imgdim(3),...
            params.NUM_CHANNELS,params.NUM_ROUNDS);
        
        tic
        for r_idx = 1:params.NUM_ROUNDS
            for c_idx = 1:params.NUM_CHANNELS
                filename = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_%s.%s',params.FILE_BASENAME,r_idx,params.SHIFT_CHAN_STRS{c_idx},regparams.REGISTRATION_TYPE,params.IMAGE_EXT));
                img = load3DImage_uint16(filename );
                img = img(crop_dims(1,1):crop_dims(1,2),...
                    crop_dims(2,1):crop_dims(2,2),...
                    crop_dims(3,1):crop_dims(3,2));
                all_seq_data(:,:,:,c_idx,r_idx) = img(y_range,x_range,:);
            end
        end
        elapsed = toc;
        fprintf('(%i, %i) All data loaded in %f seconds\n',y_idx,x_idx,elapsed );
        
        
        section_maxima = total_maxima(maxima_mask,:);
        keepers = zeros(size(section_maxima,1),1);
        fprintf('(%i, %i) Processing %i local maxima\n',y_idx,x_idx, size(section_maxima,1));
        tic
        % Note the size of the subimage we're working with
        imgdim_sub = size(all_seq_data,1:3);
        spotdim = spot_dim*2+1;
        for i=1:size(section_maxima ,1)
            
            peak = section_maxima(i,:)-offsets;
            
            [intensity, spot] = spot_caller([peak(1) peak(2) peak(3)], all_seq_data, spot_dim, corr_thresh);
            puncta_size = sum(spot(:)>corr_thresh);
            
            %Get the mean intenisty for the voxels used, which will let us
            %normalize later.
            intensity_scaled = intensity./puncta_size;
            
            %Also note the voxels used in the local area around the puncta
            %These numbers should be between 1-11
            indices_1d_local = find(spot>corr_thresh);
            %Get the XYZ around the puncta
            [y_l,x_l,z_l] = ind2sub(spotdim ,indices_1d_local); %get local pos
            %Convert those local images to the intact image size
            %Around the peak in global terms, which is the section_maxima
            [y_g,x_g,z_g] = deal(section_maxima(i,1)+ y_l-spot_dim(1)-1,...
                section_maxima(i,2)+ x_l-spot_dim(2)-1,...
                section_maxima(i,3)+ z_l-spot_dim(3)-1);%global
            indices_1d_global = sub2ind(imgdim,y_g,x_g,z_g);
            
            all_intensities(all_ctr,:,:) = intensity_scaled';
            all_sizes(all_ctr) = puncta_size;
            all_positions(all_ctr,:) = section_maxima(i,:);
            all_voxels{all_ctr} = indices_1d_global;
            all_ctr = all_ctr+1;
            
        end
        elapsed = toc;
        fprintf('(%i, %i) Done. All intensity tables created in %f seconds\n',...
            y_idx,x_idx,elapsed);
        
        
    end %end loop over x_idx


    %Finally, store only the data we used in the cell arrays
    cell_all_intensities{y_idx} = all_intensities(1:(all_ctr-1),:,:);
    cell_all_sizes{y_idx} = all_sizes(1:(all_ctr-1));
    cell_all_positions{y_idx} = all_positions(1:(all_ctr-1),:);
    cell_all_voxels{y_idx} = all_voxels(1:(all_ctr-1));
    
end %end loop over y_idx

%Collect all the sections that were calculated in parallel
all_intensities = zeros(size(total_maxima,1),params.NUM_ROUNDS,params.NUM_CHANNELS);
all_sizes = zeros(size(total_maxima,1),1);
all_positions = zeros(size(total_maxima,1),3);
all_voxels = cell(size(total_maxima,1),1);
all_ctr = 1;

for y_idx = 1:SPLIT_IMG_CORR
    N = size(cell_all_intensities{y_idx},1);
    all_intensities(all_ctr:(all_ctr+N-1),:,:) = cell_all_intensities{y_idx};
    all_sizes(all_ctr:(all_ctr+N-1)) = cell_all_sizes{y_idx};
    all_positions(all_ctr:(all_ctr+N-1),:,:) = cell_all_positions{y_idx};
    all_voxels(all_ctr:(all_ctr+N-1)) = cell_all_voxels{y_idx};
    all_ctr = all_ctr + N;
end

%% 
savepath = fullfile(params.punctaSubvolumeDir,sprintf('%s_igs_puncta.mat',params.FILE_BASENAME));
save(savepath, 'all_intensities','all_sizes','all_positions','crop_dims','all_voxels');



% for y_idx = 1:SPLIT_IMG_MAXIMA
%     y_range = y_indices(y_idx):y_indices(y_idx+1);
%     for x_idx= 1:SPLIT_IMG_MAXIMA
%         x_range = x_indices(x_idx):x_indices(x_idx+1);
%         
%         offsets = [y_range(1)-1,x_range(1)-1,0];
%         subimg = img(y_range,x_range,:);
%         tic
%         [Maxima,MaxPos,Minima,MinPos]=MinimaMaxima3D(subimg,1,0);
%         toc
%         %Only keep the peaks that are above a parameterized value
%         pos = MaxPos(Maxima>thresh,:);
%         
%         total_maxima(end+1:end+size(pos,1),:) = pos+ offsets;
%         fprintf('Total number of peaks for subsection %i\n',size(pos,1));
%     end
% end