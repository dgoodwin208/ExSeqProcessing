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

sem_name = sprintf('/%s.gc',getenv('USER'));
semaphore(sem_name,'open',1); % it is no effective if the semaphore is already open

%By default, we calculate the descriptor
if nargin<3
    skipDescriptor=false;
end
%However, in the case when we only want the keypoint (ie for Shape Context)
%we skip the calclation of the SIFT descriptor to save time

LoadParams;
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
map = ones(sift_params.image_size0, sift_params.image_size1, sift_params.image_size2, 'int8');
for i=1:N
    map(keypts(i, 1), keypts(i, 2), keypts(i, 3)) = int8(0); % select the keypoint element
end

while true
    ret = semaphore(sem_name,'trywait');
    if ret == 0
        break;
    else
        pause(1);
    end
end

fprintf('Start SIFT3D cuda calculation\n', N);
tic;
keys = sift_cuda(img, map, sift_params);
stime = toc;
fprintf('Finished SIFT3D cuda processing in %.1f s\n', stime);

ret = semaphore(sem_name,'post');

keys = num2cell(keys);

return 
