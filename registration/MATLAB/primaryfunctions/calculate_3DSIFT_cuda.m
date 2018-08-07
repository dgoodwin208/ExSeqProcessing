% Code taken from Paul Scovanner's homepage: 
% http://www.cs.ucf.edu/~pscovann/


% Calculate 3D Sift
% img is the input in question.
% keypts is an Nx3 matrix of keypoints

% returns keypoints
%   struct
%       key.x, key.y, key.z voxel indices
%       key.xyScale and key.tScale - affects both the scale and the resolution default 1
%       key.ivec descriptor vector flattened index (col order) length 
%          index shape=(sift_params.IndexSize,sift_params.IndexSize,sift_params.IndexSize,sift_params.nFaces)

function keys = calculate_3DSIFT_cuda(img, keypts,skipDescriptor)


%By default, we calculate the descriptor
if nargin<3
    skipDescriptor=false;
end
%However, in the case when we only want the keypoint (ie for Shape Context)
%we skip the calclation of the SIFT descriptor to save time

LoadParams;
save_mat = true;
image_size = size(img);
sift_params.image_size0 = image_size(1);
sift_params.image_size1 = image_size(2);
sift_params.image_size2 = image_size(3);
sift_params.skipDescriptor = skipDescriptor;

% collect fv info
fv = sphere_tri('ico', sift_params.Tessellation_levels, 1);
sift_params.fv_centers = fv.centers'; % c-order places rows contig. in memory
sift_params.fv_centers_len = length(fv.centers(:)); % default 80 * 3
assert(sift_params.fv_centers_len / 3 == sift_params.nFaces);

sift_params.keypoint_num = size(keypts,1);
N = size(keypts,1);
fprintf('SIFT3D cuda processing %d keypoints\n', N);
map = ones(sift_params.image_size0, sift_params.image_size1, sift_params.image_size2);
for i=1:N
    map(keypts(i, 1), keypts(i, 2), keypts(i, 3)) = 0; % select the keypoint element
end
unique_N = nnz(~map)
map = int8(map);

if save_mat
    assert(all(size(map) == size(img)));

    f = fopen('test_map.bin', 'w');
    fwrite(f, image_size(1), 'uint32');
    fwrite(f, image_size(2), 'uint32');
    fwrite(f, image_size(3), 'uint32');
    fwrite(f, double(map), 'double');
    fclose(f);
    fprintf('Saved test_map with %d real keypoints\n', N);

    f = fopen('test_img.bin', 'w');
    fwrite(f, image_size(1), 'uint32');
    fwrite(f, image_size(2), 'uint32');
    fwrite(f, image_size(3), 'uint32');
    fwrite(f, double(img), 'double');
    fclose(f);
    fprintf('Saved test_img \n');

    f = fopen('fv_centers.bin', 'w');
    fwrite(f, sift_params.fv_centers_len, 'uint32');
    fwrite(f, double(sift_params.fv_centers), 'double');
    fclose(f);
    fprintf('Saved fv \n');
end

keys = sift_cuda(img, map, sift_params);
keys = num2cell(keys);

return 
