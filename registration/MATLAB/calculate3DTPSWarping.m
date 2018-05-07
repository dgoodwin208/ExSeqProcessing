function calculate3DTPSWarping(moving_run)

loadParameters;
loadExperimentParams;

params.MOVING_RUN = moving_run;

disp(['[3DTPS WARP] RUNNING ON MOVING: ' num2str(params.MOVING_RUN) ', FIXED: ' num2str(params.FIXED_RUN)])

maxNumCompThreads(params.TPS3DWARP_MAX_THREADS);

affinekeys_filename = fullfile(params.OUTPUTDIR,sprintf('affinekeys_%sround%03d.h5',params.SAMPLE_NAME,params.MOVING_RUN));
if (~exist(affinekeys_filename))
    fprintf('affinekeys file was not created.\n');
    exit
end
disp('Load KeyM_total and KeyF_total that were already calculated.');
tic;
keyM_total = h5read(affinekeys_filename,'/keyM_total');
keyF_total = h5read(affinekeys_filename,'/keyF_total');
toc;

lf_shift_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_lf_sift_r%uc%u.mat',...
    params.SAMPLE_NAME,params.FIXED_RUN,1,1));
if (~exist(lf_shift_filename))
    fprintf('ShapeContext of fixed image was not created.\n');
    exit
end
load(lf_shift_filename,'img_total_size');

output_TPS_filename = fullfile(params.OUTPUTDIR,sprintf('TPSMap_%sround%03d.h5',params.SAMPLE_NAME,params.MOVING_RUN));
if ~exist(output_TPS_filename,'file')
    %        [in1D_total,out1D_total] = TPS3DWarpWhole(keyM_total,keyF_total, ...
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

