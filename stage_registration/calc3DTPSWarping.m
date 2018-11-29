function calc3DTPSWarping(moving_run,do_downsample)

loadParameters;

if do_downsample
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
end

%params.MOVING_RUN = moving_run;

fprintf('3DTPSWarp RUNNING ON MOVING: %i, FIXED: %i\n', moving_run, regparams.FIXED_RUN);

maxNumCompThreads(params.TPS3DWARP_MAX_THREADS);

affinekeys_filename = fullfile(params.registeredImagesDir,sprintf('affinekeys_%s_round%03d.h5',filename_root,moving_run));
if ~exist(affinekeys_filename)
    fprintf('affinekeys file was not created.\n');
    exit
end

filename = fullfile(params.normalizedImagesDir,sprintf('%s_round%03d_%s.%s',...
    filename_root,regparams.FIXED_RUN,regparams.CHANNELS{1},params.IMAGE_EXT ));

if isequal(params.IMAGE_EXT,'tif')
    tif_info = imfinfo(filename);
    img_total_size = [tif_info(1).Height, tif_info(1).Width, length(tif_info)];
elseif isequal(params.IMAGE_EXT,'h5')
    hdf5_info = h5info(filename,'/image')
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
end


end % function

