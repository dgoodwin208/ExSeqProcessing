function apply3DTPSWarping(moving_run,do_downsample)

loadParameters;

if do_downsample
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
end

%params.MOVING_RUN = moving_run;

fprintf('3DTPSWarpCUDA RUNNING ON MOVING: %i, FIXED: %i\n', moving_run, regparams.FIXED_RUN);

maxNumCompThreads(params.APPLY3DTPS_MAX_THREADS);


output_TPS_filename = fullfile(regparams.OUTPUTDIR,sprintf('TPSMap_%s_round%03d.h5',filename_root,moving_run));
if ~exist(output_TPS_filename,'file')
    fprintf('TPSMap file was not created.\n');
    exit
end

filename = fullfile(regparams.INPUTDIR,sprintf('%s_round%03d_%s.tif',...
    filename_root,regparams.FIXED_RUN,regparams.CHANNELS{1} ));
tif_info = imfinfo(filename);
img_total_size = [tif_info(1).Height, tif_info(1).Width, length(tif_info)];

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
    filename = fullfile(regparams.OUTPUTDIR,sprintf('%s_round%03d_%s_affine.tif',filename_root,moving_run,data_channel));
    imgToWarp = load3DTif_uint16(filename);
    toc;
    
    while true
        ret = semaphore('/gr','trywait');
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
    ret = semaphore('/gr','post');

    outputfile = fullfile(regparams.OUTPUTDIR,sprintf('%s_round%03d_%s_registered.tif',filename_root,moving_run,data_channel));
    save3DTif_uint16(outputImage_interp,outputfile);
end



end % function

