function performAffineTransforms(moving_run,do_downsample)

loadParameters;

if do_downsample
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
    image_type = 'DOWNSAMPLE';
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
    image_type = 'ORIGIMAL';
end

%params.MOVING_RUN = moving_run;

fprintf('PerfAffine RUNNING ON MOVING: %i, FIXED: %i, IMAGE TYPE: %s\n', moving_run, params.REFERENCE_ROUND_WARP,image_type);
output_affine_filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.%s',filename_root,moving_run,regparams.CHANNELS{end},params.IMAGE_EXT));
if exist(output_affine_filename,'file')
    fprintf('Already sees the last output file, skipping!\n');
    return;
end

if isfield(params,'AFFINE_MAX_THREADS')
    maxNumCompThreads(params.AFFINE_MAX_THREADS);
end

filename = fullfile(params.normalizedImagesDir,sprintf('%s_round%03d_%s.%s',...
    filename_root,params.REFERENCE_ROUND_WARP,regparams.CHANNELS{1},params.IMAGE_EXT ));

if isequal(params.IMAGE_EXT,'tif')
    tif_info = imfinfo(filename);
    img_total_size = [tif_info(1).Height, tif_info(1).Width, length(tif_info)];
elseif isequal(params.IMAGE_EXT,'h5')
    hdf5_info = h5info(filename,'/image');
    img_total_size = hdf5_info.Dataspace.Size;
else
    fprintf('unsupported file format.\n');
    exit
end

%Loading the keys, possibly from the downsampled data
output_keys_filename = fullfile(params.registeredImagesDir,sprintf('globalkeys_%s-downsample_round%03d.mat',params.FILE_BASENAME,moving_run));
if (~exist(output_keys_filename))
    fprintf('globalkeys file was not created.\n');
    exit
end

%The globalkeys file loads the keyM and keyF matrices that are used to
%calculate the warps
disp('Load KeyM_total and KeyF_total that were already calculated.');
load(output_keys_filename);

%This is quick hack as we figure out how to design the interface for when to apply the downsample. 
%If we're applying the warp that was calcualted from downsampled data, but
%we're applying it to the full-res data
if ~do_downsample && params.DO_DOWNSAMPLE
   % Scale the points of correspondence pairs back into original size
   keyM_total = keyM_total*params.DOWNSAMPLE_RATE;
   keyF_total = keyF_total*params.DOWNSAMPLE_RATE;
end

%First we do do a global affine transform on the data and keypoints before
%doing the fine-resolution non-rigid warp

%Because of the annoying switching between XY/YX conventions,
%we have to switch XY components for the affine calcs, then switch back
keyM_total_switch = keyM_total(:,[2 1 3]);
keyF_total_switch = keyF_total(:,[2 1 3]);

%The old way was calculating the affine tform
warning('off','all'); 
affine_tform = findAffineModel(keyM_total_switch, keyF_total_switch,regparams.AFFINE_FULL);
keyF_total_switch = [];
warning('on','all')

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
output_affinekeys_filename = fullfile(params.registeredImagesDir,sprintf('affinekeys_%s_round%03d.h5',filename_root,moving_run));
if exist(output_affinekeys_filename)
    delete(output_affinekeys_filename);
end
h5create(output_affinekeys_filename,'/keyM_total',size(keyM_total));
h5create(output_affinekeys_filename,'/keyF_total',size(keyF_total));
h5write(output_affinekeys_filename,'/keyM_total',keyM_total);
h5write(output_affinekeys_filename,'/keyF_total',keyF_total);
toc;

ch_list = regparams.CHANNELS;
input_chandir = params.colorCorrectionImagesDir;
input_normdir = params.normalizedImagesDir;
outputdir = params.registeredImagesDir;
image_ext = params.IMAGE_EXT;
if isfield(params, 'MORPHOLOGY_ROUND') && (moving_run == params.MORPHOLOGY_ROUND)
    ch_list{end+1} = [params.MORPHOLOGY_CHAN_STR,'SHIFT'];
end
if isfield(params,'AFFINE_MAX_THREADS')
    worker_max_threads = params.AFFINE_MAX_THREADS;
else
    worker_max_threads = 'automatic';
end
parfor c = 1:length(ch_list)
    maxNumCompThreads(worker_max_threads);
    %Load the data to be warped
    tic;
    data_channel = ch_list{c};
    fprintf('load 3D file for affine transform on %s channel\n',data_channel);
    if contains(data_channel,'ch')
        inputdir = input_chandir;
    else
        inputdir = input_normdir;
    end
    filename = fullfile(inputdir,sprintf('%s_round%03d_%s.%s',filename_root,moving_run,data_channel,image_ext));
    imgToWarp = load3DImage_uint16(filename);
    toc;

    output_affine_filename = fullfile(outputdir,sprintf('%s_round%03d_%s_affine.%s',...
        filename_root,moving_run,data_channel,image_ext));

    imgMoving_total_affine = imwarp(imgToWarp,affine3d(affine_tform'),'OutputView',rF);
    save3DImage_uint16(imgMoving_total_affine,output_affine_filename);
end



end % function

