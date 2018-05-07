function apply3DTPSWarping(moving_run)

loadParameters;
loadExperimentParams;

params.MOVING_RUN = moving_run;

disp(['[APPLY 3DTPS] RUNNING ON MOVING: ' num2str(params.MOVING_RUN) ', FIXED: ' num2str(params.FIXED_RUN)])

maxNumCompThreads(params.APPLY3DTPS_MAX_THREADS);


output_TPS_filename = fullfile(params.OUTPUTDIR,sprintf('TPSMap_%sround%03d.h5',params.SAMPLE_NAME,params.MOVING_RUN));
if exist(output_TPS_filename,'file')==0
    fprintf('TPSMap file was not created.\n');
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

lf_shift_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_lf_sift_r%uc%u.mat',...
    params.SAMPLE_NAME,params.FIXED_RUN,1,1));
if (~exist(lf_shift_filename))
    fprintf('ShapeContext of fixed image was not created.\n');
    exit
end
load(lf_shift_filename,'img_total_size');

%Warp all three channels of the experiment once the index mapping has been
%created
for c = 1:length(params.CHANNELS)
    %Load the data to be warped
    disp('load 3D file to be warped')
    tic;
    data_channel = params.CHANNELS{c};
    filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_%s_affine.tif',params.SAMPLE_NAME,params.MOVING_RUN,data_channel));
    imgToWarp = load3DTif_uint16(filename);
    toc;
    
    %we loaded the bounds_moving data at the very beginning of this file
    cropfilename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_cropbounds.mat',params.SAMPLE_NAME,params.MOVING_RUN));
    if exist(cropfilename,'file')==2
        load(cropfilename,'bounds'); bounds_moving = floor(bounds); clear bounds;
        imgToWarp = imgToWarp(bounds_moving(1):bounds_moving(2),bounds_moving(3):bounds_moving(4),:);
    end

    while true
        ret = semaphore('/gr','trywait');
        if ret == 0
            break;
        else
            pause(1);
        end
    end
    t_tps3dapply = tic;
    [ outputImage_interp ] = TPS3DApply(in1D_total,out1D_total,imgToWarp,img_total_size,data_channel);
    fprintf('transform image with 3DTPS in %s channel. ',data_channel);
    toc(t_tps3dapply);
    ret = semaphore('/gr','post');

    outputfile = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_%s_registered.tif',params.SAMPLE_NAME,params.MOVING_RUN,data_channel));
    save3DTif_uint16(outputImage_interp,outputfile);
end



end % function

