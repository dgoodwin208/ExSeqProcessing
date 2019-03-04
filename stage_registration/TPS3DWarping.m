function TPS3DWarping(moving_run,do_downsample)

loadParameters;

if do_downsample
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
    image_type = 'DOWNSAMPLE';
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
    image_type = 'ORIGIMAL';
end

fprintf('TPS3DWarp RUNNING ON MOVING: %i, FIXED: %i, IMAGE TYPE: %s\n', moving_run, params.REFERENCE_ROUND_WARP,image_type);

if isfield(params,'TPS3DWARP_MAX_THREADS')
    maxNumCompThreads(params.TPS3DWARP_MAX_THREADS);
end

affinekeys_filename = fullfile(params.registeredImagesDir,sprintf('affinekeys_%s_round%03d.h5',filename_root,moving_run));
if ~exist(affinekeys_filename)
    fprintf('affinekeys file was not created.\n');
    exit
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

disp('Load KeyM_total and KeyF_total that were already calculated.');
tic;
keyM_total = h5read(affinekeys_filename,'/keyM_total');
keyF_total = h5read(affinekeys_filename,'/keyF_total');
toc;

output_TPS_filename = fullfile(params.registeredImagesDir,sprintf('TPSMap_%s_round%03d.h5',filename_root,moving_run));
if ~exist(output_TPS_filename,'file')

    %These keypoints have now been modified by the affine warp, so are in
    %the dimensinos of the keyFTotal
    [in1D_total,out1D_total] = TPS3DWarpWholeInParallel(keyM_total,keyF_total, img_total_size, img_total_size);
    disp('save TPS file as hdf5')
    tic;
    h5create(output_TPS_filename,'/in1D_total',size(in1D_total));
    h5create(output_TPS_filename,'/out1D_total',size(in1D_total));
    h5write(output_TPS_filename,'/in1D_total',in1D_total);
    h5write(output_TPS_filename,'/out1D_total',out1D_total);
    toc;
else
    %load in1D_total and out1D_total
    disp('load TPS file as hdf5')
    tic;
    in1D_total = h5read(output_TPS_filename,'/in1D_total');
    out1D_total = h5read(output_TPS_filename,'/out1D_total');
    toc;
    %Experiments 7 and 8 may have been saved with zeros in the 1D vectors
    %so this removes it
    [ValidIdxs,I] = find(in1D_total>0);
    in1D_total = in1D_total(ValidIdxs);
    out1D_total = out1D_total(ValidIdxs);
end


%Warp all three channels of the experiment once the index mapping has been
%created
for c = 1:length(regparams.CHANNELS)
    %Load the data to be warped
    disp('load 3D file to be warped')
    tic;
    data_channel = regparams.CHANNELS{c};
    filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.%s',filename_root,moving_run,data_channel,params.IMAGE_EXT));
    imgToWarp = load3DImage_uint16(filename);
    toc;

    t_tps3dapply = tic;
    [ outputImage_interp ] = TPS3DApplyCUDA(in1D_total,out1D_total,imgToWarp,img_total_size,data_channel);
    fprintf('transform image with 3DTPS in %s channel. ',data_channel);
    toc(t_tps3dapply);

    outputfile = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_registered.%s',filename_root,moving_run,data_channel,params.IMAGE_EXT));
    save3DImage_uint16(outputImage_interp,outputfile);
end


end % function

