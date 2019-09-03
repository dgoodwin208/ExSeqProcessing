
ROUNDS = 1:params.NUM_ROUNDS;
if isfield(params, 'MORPHOLOGY_ROUND') && (params.MORPHOLOGY_ROUND <= params.NUM_ROUNDS)
    ROUNDS(params.MORPHOLOGY_ROUND) = [];
end
for roundnum = ROUNDS


    summed_norm = load3DImage_uint16(fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_%s.%s',params.FILE_BASENAME,roundnum,'summedNorm',regparams.REGISTRATION_TYPE,params.IMAGE_EXT)));

    %NEW: 08/01/2018: high pass filter before segmentation
    %fprintf('Creating GPU array\n');
    %summed_norm_gpu = gpuArray(single(summed_norm));
    %fprintf('Bluring\n');
    %img_blur_gpu = imgaussfilt3(summed_norm_gpu,[30 30 30*(.17/.4)]);
    %fprintf('Gathering\n');
    %img_blur = gather(img_blur_gpu);
    %catch
%	fprintf('Failed to load round %i\n',roundnum)
%        continue;
%    end
    %remove any zeros by just putting the mean of the data in:
    %summed_norm(summed_norm==0) = mean(summed_norm(summed_norm>0));
    if ~exist('total_summed_norm','var')
        total_summed_norm = summed_norm;
    else
        total_summed_norm = total_summed_norm + summed_norm;
        %geometric meean exploration:
        %total_summed_norm = total_summed_norm.*summed_norm;
    end

end

%If geometric mean
%total_summed_norm = total_summed_norm.^(1/params.NUM_ROUNDS);

min_total = min(total_summed_norm(:));
total_summed_norm_scaled = total_summed_norm - min_total;
total_summed_norm_scaled = (total_summed_norm_scaled/max(total_summed_norm_scaled(:)))*double(intmax('uint16'));
total_summed_norm = [];
save3DImage_uint16(total_summed_norm_scaled,fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm.%s',params.FILE_BASENAME,params.IMAGE_EXT)));

%Note the original size of the data before upscaling to isotropic voxels
data = total_summed_norm_scaled;
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

pool_size = 10;
delete(gcp('nocreate'))
parpool(pool_size);

data_size1 = size(data,1);
data_size2 = size(data,2);
dy = uint32(data_size1 / pool_size);

len_data_interp = length(data_interp);

y_starts = zeros(pool_size,1);
y_ends = zeros(pool_size,1);
for i = 1:pool_size
    y_starts(i) = 1 + dy*(i-1);
    y_ends(i) = dy*i;
end
if y_ends(end) > data_size1
    y_ends(end) = data_size1;
end
y_elms = (y_ends-y_starts)+1;

data_interpolated_cell = cell(pool_size);
data_cell = mat2cell(data,y_elms);
data = [];

%%Interpolate using pieceswise cubic interpolation,
%%pchip and cubic actually are the same, according to MATLAB doc in 2016b
parfor i = 1:pool_size
    data_interpolated_cell{i} = zeros(y_elms(i),data_size2,len_data_interp);
    for y = 1:y_elms(i)
        for x = 1:data_size2
            data_interpolated_cell{i}(y,x,:) = interp1(indices_orig,squeeze(data_cell{i}(y,x,:)),query_pts_interp,'pchip');
        end
    end
end
data_cell = {};

for i = 1:pool_size
    data_interpolated(y_starts(i):y_ends(i),:,:) = data_interpolated_cell{i};
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
len_data_interp = length(data_interp);

data_interp = [];

data_interpolated_cell = cell(pool_size);
data_cell = mat2cell(data,y_elms);
data = [];

parfor i = 1:pool_size
    data_interpolated_cell{i} = zeros(y_elms(i),data_size2,len_data_interp);
    for y = 1:y_elms(i)
        for x = 1:data_size2
            %'Nearest' = nearest neighbor, means there should be no new values
            %being created
            data_interpolated_cell{i}(y,x,:) = interp1(query_pts_interp,squeeze(data_cell{i}(y,x,:)),indices_orig,'nearest');
        end
    end
end
data_cell = {};

for i = 1:pool_size
    data_interpolated(y_starts(i):y_ends(i),:,:) = data_interpolated_cell{i};
end

L_origsize = uint32(data_interpolated);
data_interpolated = [];

%Extract the puncta from the watershed output (no longer interpolated)
%params.PUNCTA_SIZE_THRESHOLD = 30;
%params.PUNCTA_SIZE_MAX = 2000;

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
save(filename_centroids, 'puncta_centroids','puncta_voxels', '-v7.3');
