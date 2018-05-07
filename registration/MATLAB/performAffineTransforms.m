function performAffineTransforms(moving_run)

loadParameters;
loadExperimentParams;

params.MOVING_RUN = moving_run;

disp(['[AFFINE] RUNNING ON MOVING: ' num2str(params.MOVING_RUN) ', FIXED: ' num2str(params.FIXED_RUN)])

maxNumCompThreads(params.AFFINE_MAX_THREADS);

output_keys_filename = fullfile(params.OUTPUTDIR,sprintf('globalkeys_%sround%03d.mat',params.SAMPLE_NAME,params.MOVING_RUN));
if (~exist(output_keys_filename))
    fprintf('globalkeys file was not created.\n');
    exit
end
disp('Load KeyM_total and KeyF_total that were already calculated.');
load(output_keys_filename);

lf_shift_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_lf_sift_r%uc%u.mat',...
    params.SAMPLE_NAME,params.FIXED_RUN,1,1));
if (~exist(lf_shift_filename))
    fprintf('ShapeContext of fixed image was not created.\n');
    exit
end
load(lf_shift_filename,'img_total_size');

fprintf('Using all %i corresondences by ignoring for quantile cutoff\n', size(keyM_total,1));

%If you want to set a maxium warp distance between matching keypoints. This is not currently being used but keeping it in mainly as a reminder that the correspondeces were problematic in the past and code like this can be implemented if need be.
if (params.MAXDISTANCE>-1)
    remove_indices = [];
    for match_idx = 1:size(keyF_total,1)
        if norm(keyF_total(match_idx,:)-keyM_total(match_idx,:))>params.MAXDISTANCE
            norm(keyF_total(match_idx,:)-keyM_total(match_idx,:));
            remove_indices = [remove_indices match_idx];
        end
    end
    keyF_total(remove_indices,:) = [];
    keyM_total(remove_indices,:) = [];

    clear remove_indices;
end

if isempty(keyF_total) || isempty(keyM_total)
    error('ERROR: all keys removed, consider raising `params.MAXDISTANCE`... exiting');
end

%Do a global affine transform on the data and keypoints before
%doing the fine-resolution non-rigid warp

%Because of the annoying switching between XY/YX conventions,
%we have to switch XY components for the affine calcs, then switch back
keyM_total_switch = keyM_total(:,[2 1 3]);
keyF_total_switch = keyF_total(:,[2 1 3]);

%The old way was calculating the affine tform
warning('off','all'); 
affine_tform = findAffineModel(keyM_total_switch, keyF_total_switch);
keyF_total_switch = [];
warning('on','all')

if ~det(affine_tform)
    error('ERROR: affine_tform can not be singular for following calcs... exiting')
end

%Warp the keyM features into the new space
rF = imref3d(img_total_size);
%Key total_affine is now with the switched XY
keyM_total_affine = [keyM_total_switch, ones(size(keyM_total_switch,1),1)]*affine_tform';
keyM_total_switch = [];
%keyM_total is now switched
keyM_total=keyM_total_affine(:,1:3);
%keyF_total = keyF_total_switch;
%Remove any keypoints which are now outside the bounds of the image
filtered_correspondence_indices = (keyM_total(:,1) <1 | keyM_total(:,2)<1 | keyM_total(:,3)<1 | ...
    keyM_total(:,1) > img_total_size(2) | ...
    keyM_total(:,2) > img_total_size(1) | ...
    keyM_total(:,3) > img_total_size(3) );
fprintf('Losing %i features after affine warp\n',sum(filtered_correspondence_indices));
keyM_total(filtered_correspondence_indices,:) = [];
keyF_total(filtered_correspondence_indices,:) = [];

%switch keyM back to the other format for the TPS calcs
keyM_total = keyM_total(:,[2 1 3]);

disp('save affine-transformed keys file as hdf5')
tic;
output_affinekeys_filename = fullfile(params.OUTPUTDIR,sprintf('affinekeys_%sround%03d.h5',params.SAMPLE_NAME,params.MOVING_RUN));
if exist(output_affinekeys_filename)
    delete(output_affinekeys_filename);
end
h5create(output_affinekeys_filename,'/keyM_total',size(keyM_total));
h5create(output_affinekeys_filename,'/keyF_total',size(keyF_total));
h5write(output_affinekeys_filename,'/keyM_total',keyM_total);
h5write(output_affinekeys_filename,'/keyF_total',keyF_total);
toc;

parfor c = 1:length(params.CHANNELS)
    %Load the data to be warped
    tic;
    data_channel = params.CHANNELS{c};
    fprintf('load 3D file for affine transform on %s channel\n',data_channel);
    filename = fullfile(params.INPUTDIR,sprintf('%sround%03d_%s.tif',params.SAMPLE_NAME,params.MOVING_RUN,data_channel));
    imgToWarp = load3DTif_uint16(filename);
    toc;

%    imgMoving_total_size = size(imgToWarp);

    output_affine_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_%s_affine.tif',...
        params.SAMPLE_NAME,params.MOVING_RUN,data_channel));
    imgMoving_total_affine = imwarp(imgToWarp,affine3d(affine_tform'),'OutputView',rF);
    save3DTif_uint16(imgMoving_total_affine,output_affine_filename);
end



end % function

