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
sift_params.descriptor_len = sift_params.IndexSize *...
    sift_params.IndexSize * sift_params.IndexSize * sift_params.nFaces;

sift_params.keypoint_num = size(keypts,1);
N = size(keypts,1);
fprintf('SIFT3D cuda processing %d keypoints\n', N);
map = ones(sift_params.image_size0, sift_params.image_size1, sift_params.image_size2);
for i=1:N
    map(keypts(i, 1), keypts(i, 2), keypts(i, 3)) = 0; % select the keypoint element
end
map = int8(map);

%fprintf('Save map with %d real keypoints\n', N);
%f = fopen('map_2kypts.bin', 'w');
%fwrite(f, map);
%fclose(f);

%fprintf('Save img \n');
%f = fopen('img_2kypts.bin', 'w');
%fwrite(f, img);
%fclose(f);

keys = sift_cuda(img, map, sift_params);
keys = num2cell(keys);


return 

%i = 0;
%offset = 0;
%precomp_grads = {};
%precomp_grads.count = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3));
%precomp_grads.mag = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3));
%precomp_grads.ix = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), ...
    %sift_params.Tessel_thresh, 1);
%precomp_grads.yy = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), ...
    %sift_params.Tessel_thresh, 1);
%precomp_grads.vect = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), 1, 3);
%while 1

    %reRun = 1;
    %i = i+1;
    
    %while reRun == 1
        
        %loc = keypts(i+offset,:);
        %%fprintf(1,'Calculating keypoint at location (%d, %d, %d)\n',loc);
        
        %% Create a 3DSIFT descriptor at the given location
        %if ~skipDescriptor
            %[keys{i} reRun precomp_grads] = Create_Descriptor(img,1,1,loc(1),loc(2),loc(3),sift_params, precomp_grads);
        %else         
            %clear k; reRun=0;
            %k.x = loc(1); k.y = loc(2); k.z = loc(3);
            %keys{i} = k;
        %end

        %if reRun == 1
            %offset = offset + 1;
        %end
        
        %%are we out of data?
        %if i+offset>=size(keypts,1)
            %break;
        %end
    %end
    
    %%are we out of data?
    %if i+offset>=size(keypts,1)
            %break;
    %end
%end

%% for diagnostics only, runs extremely slow
%%if ~skipDescriptor
    %%tic
    %%recomps = 0;
    %%value_arr = values(precomp_grads);
    %%for i=1:length(value_arr)
        %%val_cell = value_arr{i};
        %%if val_cell.count > 1
            %%recomps = recomps + val_cell.count - 1;
        %%end
    %%end
    %%fprintf(1, '\n%d redundant pixels of %d total visited, saving 3DSIFTkeys\n', recomps, length(precomp_grads));
    %%save('precomp_grads', 'precomp_grads');
    %%save('3DSIFTkeys', 'keys');
    %%toc
%%end

%fprintf(1,'\nFinished.\n%d points thrown out do to poor descriptive ability.\n',offset);

%end
