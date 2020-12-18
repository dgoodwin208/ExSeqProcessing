function performAffineTransforms_global(fixed_fov, moving_fovmatches, moving_round,bigparams)


filename_root = sprintf('%s-F%.3i',bigparams.EXPERIMENT_NAME,fixed_fov);
image_type = 'ORIGINAL';


ch_list = bigparams.CHANNELS;
image_ext = bigparams.IMAGE_EXT;
outputdir = fullfile(bigparams.EXPERIMENT_FOLDERROOT,...
    sprintf('F%.3i',fixed_fov),...
    '4_registration');


fprintf('PerfAffine RUNNING ON MOVING: %i, FIXED: %i, IMAGE TYPE: %s\n', moving_round, bigparams.REFERENCE_ROUND,image_type);

% commented only because I don't know what this is doing
% if isfield(params,'AFFINE_MAX_THREADS')
%     maxNumCompThreads(params.AFFINE_MAX_THREADS);
% end

input_normdir = fullfile(bigparams.EXPERIMENT_FOLDERROOT,...
   sprintf('F%.3i',fixed_fov),'3_normalization');
filename = fullfile(input_normdir,sprintf('%s_round%03d_%s.%s',...
    filename_root,bigparams.REFERENCE_ROUND,bigparams.CHANNELS{1},image_ext));

if isequal(image_ext ,'tif')
    tif_info = imfinfo(filename);
    img_total_size = [tif_info(1).Height, tif_info(1).Width, length(tif_info)];
elseif isequal(image_ext ,'h5')
    hdf5_info = h5info(filename,'/image');
    img_total_size = hdf5_info.Dataspace.Size;
else
    fprintf('unsupported file format.\n');
    exit
end
%Warp the keyM features into the new space
rF = imref3d(img_total_size);

fprintf('Calculating the warp for FOV%.3i will be using fovs from: %s\n',fixed_fov,mat2str(moving_fovmatches));


%initialize the output image
outputimg_allchans= zeros(length(ch_list), img_total_size(1),img_total_size(2),...
        img_total_size(3));


keys_fixed = loadFOVKeyptsAndFeatures(fixed_fov,bigparams.REFERENCE_ROUND,bigparams);
for fov_mov = moving_fovmatches
    
    %Load the (nonDOWNSAMPLED) keypoints for the moving and fixed fovs, all
    %local coordinates for their own FOV
    keys_moving = loadFOVKeyptsAndFeatures(fov_mov,moving_round,bigparams);
    if isempty(keys_moving) || isempty(keys_fixed)
        fprintf('ERROR: missing a keys file\n');
        continue
    end
    
    try
        [keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed);
        %Calculate the affine tform and get back the transformed keypoints
        [affine_tform,keyM_total_tformed]  = getGlobalAffineFromCorrespondences(keyM_total,keyF_total);
    catch
        fprintf("failed to find sufficient correspondences between FOVS %i and %i, skip!\n",FOV_fixed,fov_mov)
        continue
    end
    affine_tform
    %if it's a flimsy tform post-ransac, better to leave it blank
    if size(keyM_total_tformed,1)<3
        continue
    end
%     if isfield(params,'AFFINE_MAX_THREADS')
%         worker_max_threads = params.AFFINE_MAX_THREADS;
%     else
%         worker_max_threads = 'automatic';
%     end
%     maxNumCompThreads(worker_max_threads);
    
    
    
    filename_root_moving = sprintf('%s-F%.3i',bigparams.EXPERIMENT_NAME,fov_mov);
    input_chandir = fullfile(bigparams.EXPERIMENT_FOLDERROOT,...
         sprintf('F%.3i',fov_mov),'2_color-correction');    
    input_normdir = fullfile(bigparams.EXPERIMENT_FOLDERROOT,...
         sprintf('F%.3i',fov_mov),'3_normalization');
    for c = 1:length(ch_list)
        %Load the data to be warped
        data_channel = ch_list{c};
        fprintf('load 3D file for affine transform on %s channel\n',data_channel);
        if contains(data_channel,'ch')
            inputdir = input_chandir;
        else
            inputdir = input_normdir;
        end

        filename = fullfile(inputdir,sprintf('%s_round%03d_%s.%s',filename_root_moving,moving_round,data_channel,image_ext));
        imgToWarp = load3DImage_uint16(filename);
        
        
        
        imgMoving_total_affine = imwarp(imgToWarp,affine3d(affine_tform'),'OutputView',rF);
        %Do element-wise maximum for this moving fov, which may just be
        %part of the fixed fov
        outputimg_allchans(c,:,:,:) = max(squeeze(outputimg_allchans(c,:,:,:)),imgMoving_total_affine);
    end  %finish looping over colors
end %finish looping over moving rounds

%save the final 
for c = 1:length(ch_list)
    data_channel = ch_list{c};
    output_affine_filename = fullfile(outputdir,sprintf('%s_round%03d_%s_affine.%s',...
        filename_root,moving_round,data_channel,image_ext));
    outputimg = squeeze(outputimg_allchans(c,:,:,:));
    save3DImage_uint16(outputimg,output_affine_filename);
    
    %Also just downsample too 
    outputimg_ds = imresize3(outputimg,1/bigparams.DOWNSAMPLE_RATE,'linear');
    output_affine_filename = fullfile(outputdir,sprintf('%s-downsample_round%03d_%s_affine.%s',...
        filename_root,moving_round,data_channel,image_ext));
    
    save3DImage_uint16(outputimg_ds,output_affine_filename);

end


end % function

