function apply3DTPSWarping(moving_run,do_downsample)

loadParameters;
sem_name = sprintf('/%s.gr',getenv('USER'));

if do_downsample
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
end

%params.MOVING_RUN = moving_run;

fprintf('3DTPSWarpCUDA RUNNING ON MOVING: %i, FIXED: %i\n', moving_run, regparams.FIXED_RUN);

maxNumCompThreads(params.APPLY3DTPS_MAX_THREADS);


output_TPS_filename = fullfile(params.registeredImagesDir,sprintf('TPSMap_%s_round%03d.h5',filename_root,moving_run));
if ~exist(output_TPS_filename,'file')
    fprintf('TPSMap file was not created.\n');
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
    
    while true
        ret = semaphore(sem_name,'trywait');
        if ret == 0
            break;
        else
            pause(1);
        end
    end
    t_tps3dapply = tic;
    [ outputImage_interp ] = TPS3DApplyCUDA(in1D_total,out1D_total,imgToWarp,img_total_size,data_channel);
    fprintf('transform image with 3DTPS in %s channel. ',data_channel);
    toc(t_tps3dapply);
    ret = semaphore(sem_name,'post');

    outputfile = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_registered.%s',filename_root,moving_run,data_channel,params.IMAGE_EXT));
    save3DImage_uint16(outputImage_interp,outputfile);
end



end % function

