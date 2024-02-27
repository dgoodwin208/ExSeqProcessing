% Code taken from Paul Scovanner's homepage: 
% http://www.cs.ucf.edu/~pscovann/


% Calculate 3D Sift
% img is the input in question.
% keypts is an Nx3 matrix of keypoints

function  keys = calculate_3DSIFT(img, keypts,skipDescriptor)

%By default, we calculate the descriptor
if nargin<3
    skipDescriptor=false;
end
%However, in the case when we only want the keypoint (ie for Shape Context)
%we skip the calclation of the SIFT descriptor to save time

loadParameters;
sift_params.pix_size = size(img);
keys = cell(size(keypts,1),1);
precomp_grads = {};
precomp_grads.count = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3));
precomp_grads.mag = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3));
precomp_grads.ix = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), ...
    sift_params.Tessel_thresh, 1);
precomp_grads.yy = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), ...
    sift_params.Tessel_thresh, 1);
precomp_grads.vect = zeros(sift_params.pix_size(1), sift_params.pix_size(2), sift_params.pix_size(3), 1, 3);

key_idx = 1;
skipped = 0;
for i = 1:size(keypts,1)

    loc = keypts(i,:);
    %fprintf(1,'Calculating keypoint at location (%d, %d, %d)\n',loc);

    % Create a 3DSIFT descriptor at the given location
    if ~skipDescriptor
        [kpt reRun precomp_grads] = Create_Descriptor(img,sift_params.xyScale,sift_params.tScale,loc(1),loc(2),loc(3),sift_params, precomp_grads);
        if reRun == 0
            keys{key_idx} = kpt;
            key_idx = key_idx + 1;
        else
            skipped = skipped + 1;
        end
    else
        clear k
        k.x = loc(1); k.y = loc(2); k.z = loc(3);
        keys{key_idx} = k;
        key_idx = key_idx + 1;
    end

end
%remove any pre-initialized descriptors that weren't used
keys(key_idx:end)=[];

% for diagnostics only, warning: runs extremely slowly
%if ~skipDescriptor
    %tic
    %recomps = 0;
    %value_arr = values(precomp_grads);
    %for i=1:length(value_arr)
        %val_cell = value_arr{i};
        %if val_cell.count > 1
            %recomps = recomps + val_cell.count - 1;
        %end
    %end
    %fprintf(1, '\n%d redundant pixels of %d total visited, saving 3DSIFTkeys\n', recomps, length(precomp_grads));
    %save('precomp_grads', 'precomp_grads');
    %save('3DSIFTkeys', 'keys');
    %toc
%end

fprintf(1,'\nFinished.\n%d points thrown out do to poor descriptive ability.\n',skipped);

end
