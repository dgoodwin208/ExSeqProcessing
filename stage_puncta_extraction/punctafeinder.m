
ROUNDS = 1:params.NUM_ROUNDS;
for roundnum = ROUNDS
    

    summed_norm = load3DImage_uint16(fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_%s.%s',params.FILE_BASENAME,roundnum,params.PUNCTA_CHANNEL_SEG,regparams.REGISTRATION_TYPE,params.IMAGE_EXT)));

    if ~exist('total_summed_norm','var')
        total_summed_norm = summed_norm;
        
        %IN BRANCH: The zero_mask_tracker will count 0 values, which is the
        %indicator of being outside of the registered volume
        zero_mask_tracker = zeros(size(total_summed_norm));
        zero_mask_tracker = zero_mask_tracker + (total_summed_norm==0);
    else
        total_summed_norm = total_summed_norm + summed_norm;
        zero_mask_tracker = zero_mask_tracker + (total_summed_norm==0);
        %geometric meean exploration:
        %total_summed_norm = total_summed_norm.*summed_norm;
    end

end


min_total = min(total_summed_norm(:));
total_summed_norm_scaled = total_summed_norm - min_total;
total_summed_norm_scaled = (total_summed_norm_scaled/max(total_summed_norm_scaled(:)))*double(intmax('uint16'));
total_summed_norm = [];
save3DImage_uint16(total_summed_norm_scaled,fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm.%s',params.FILE_BASENAME,params.IMAGE_EXT)));

% Convert the zero_mask into a cropping bounds
mask = zero_mask_tracker<=param.MAXNUM_MISSINGROUND;
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
total_summed_norm_scaled = total_summed_norm_scaled(...
    crop_dims(1,1):crop_dims(1,2),...
    crop_dims(2,1):crop_dims(2,2),...
    crop_dims(3,1):crop_dims(3,2));
    
save3DImage_uint16(total_summed_norm_scaled,fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_cropped.%s',params.FILE_BASENAME,params.IMAGE_EXT)));

%Note the original size of the data before upscaling to isotropic voxels
data = total_summed_norm_scaled;

%Using feedback from Eftychios Pnevmatatiks from the Flatiron Institute, we add bg substraction
fprintf('Starting new background subtraction...');
%New background subtraction method: Added 20191009
se = strel('sphere',params.PUNCTARADIUS_BGESTIMATE);
data_opened = imopen(data,se);
data = max(data - data_opened,0);
fprintf('Done!\n');
%end bg subtraction

total_summed_norm_scaled = [];
img_origsize = data;

%This requires knowledge of the Z and XY resolutions
%Z_upsample_factor = .5/.175;
Z_upsample_factor = params.ZRES/params.XRES;

indices_orig = 1:size(data,3); %Note the indices of the original size of the image
query_pts_interp = 1:1/Z_upsample_factor:size(data,3); %Query points for interpolation

%Do the opposite of the interpolation
data_interp = interp1(indices_orig,squeeze(data(1,1,:)),query_pts_interp);
data_interpolated = zeros(size(data,1),size(data,2),length(data_interp));

%Interpolate using pieceswise cubic interpolation, 
%pchip and cubic actually are the same, according to MATLAB doc in 2016b
for y = 1:size(data,1)
    for x = 1:size(data,2)
        if all(squeeze(data(y,x,:))==0)
            %For speed, if the whole z-column is 0, no need to interpolate!
            data_interpolated(y,x,:)=0;
        else
            data_interpolated(y,x,:) = interp1(indices_orig,squeeze(data(y,x,:)),query_pts_interp,'pchip');
        end
    end
    
    if mod(y,200)==0
        fprintf('\t%i/%i rows interpolated\n',y,size(data,1));
    end
end

%Now use the punctafeinder's dog filtering approach
stack_in = data_interpolated;
%stack_orig = stack_in;
stack_in = dog_filter(stack_in);

%Thresholding is still the most sensitive part of this pipeline, so results
%must be analyzed with this two lines of code in mind
thresh = multithresh(stack_in(stack_in>0),1);
fgnd_mask = stack_in>thresh(1);

%Ignore the background via thresholding and calculate watershed
stack_in(~fgnd_mask) = 0; % thresholded using dog background
fgnd_mask = [];
neg_masked_image = -int32(stack_in);
neg_masked_image(~stack_in) = inf;

tic
fprintf('Watershed running...');
L = uint32(watershed(neg_masked_image));
neg_masked_image = [];
L(~stack_in) = 0;
stack_in = [];
fprintf('DONE\n');
toc

%Now we downsample the output of the watershed, L, back into the original
%space
data = double(L); %interpolation needs double or single, L is integer
L = [];
data_interp = interp1(query_pts_interp,squeeze(data(1,1,:)),indices_orig);
data_interpolated = zeros(size(data,1),size(data,2),length(data_interp));
data_interp = [];

for y = 1:size(data,1)
    %TODO: We could skip the vectors that are all zero from interpolation
    %like we did above.
    for x = 1:size(data,2)

        if all(squeeze(data(y,x,:))==0)
	    %For speed, if the whole z-column is 0, no need to interpolate!
            data_interpolated(y,x,:)=0;
        else
            %'Nearest' = nearest neighbor, means there should be no new values
            %being created
            data_interpolated(y,x,:) = interp1(query_pts_interp,squeeze(data(y,x,:)),indices_orig,'nearest');
        end
    end
    
    if mod(y,200)==0
        fprintf('\t%i/%i rows processed\n',y,size(data,1));
    end
end
L_origsize = uint32(data_interpolated);
data_interpolated = [];
%clear data to make space
data = [];
%Extract the puncta from the watershed output (no longer interpolated)
candidate_puncta= regionprops(L_origsize,img_origsize, 'WeightedCentroid', 'PixelIdxList');
L_origsize = [];
indices_to_remove = [];
for i= 1:length(candidate_puncta)
    if size(candidate_puncta(i).PixelIdxList,1)< params.PUNCTA_SIZE_THRESHOLD ...
        || size(candidate_puncta(i).PixelIdxList,1)> params.PUNCTA_SIZE_MAX
        indices_to_remove = [indices_to_remove i];
    end
end

good_indices = 1:length(candidate_puncta);
good_indices(indices_to_remove) = [];

%A commmon failure of watershed is that the threshold is too low 
%for now, we're going to hardcode 
max_puncta = numel(img_origsize)/(params.PUNCTA_SIZE^3)
if length(good_indices)>max_puncta
    error(sprintf('Way too many puncta extracted: %i puncta with a threshold of %f\n',length(good_indices),thresh))
end
filtered_puncta = candidate_puncta(good_indices);

output_img = zeros(size(img_origsize));
img_origsize = [];

for i= 1:length(filtered_puncta)
    %Set the pixel value to somethign non-zero
    output_img(filtered_puncta(i).PixelIdxList)=100;
end

filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
save3DImage_uint16(output_img,filename_out);
output_img = [];

%Creating the list of voxels and centroids per round is entirely
%unnecessary, but keeping it in for now for easy of processing
%for rnd_idx=1:params.NUM_ROUNDS
    num_puncta_per_round = length(filtered_puncta);
    
    %initialize the vectors for the particular round
    voxels_per_round = cell(num_puncta_per_round,1);
    centroids_per_round = zeros(num_puncta_per_round,3);

    for ctr = 1:num_puncta_per_round
        %Note the indices of this puncta
        voxels_per_round{ctr} = filtered_puncta(ctr).PixelIdxList;
        centroids_per_round(ctr,:) = filtered_puncta(ctr).WeightedCentroid;
    end
    
    puncta_centroids = centroids_per_round;
    puncta_voxels = voxels_per_round;
%end

filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
save(filename_centroids, 'puncta_centroids','puncta_voxels','crop_dims', '-v7.3');

