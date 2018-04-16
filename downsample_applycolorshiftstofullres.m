loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    return;
end

for rnd_indx = 1:params.NUM_ROUNDS
    

    chan1_inname = fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch00.tif',params.FILE_BASENAME,rnd_indx));
    if ~exist(chan1_inname,'file')
        fprintf('Skipping missing round %i \n',rnd_indx);
        continue
    end
    
    filename_colorShifts = fullfile(OUTPUTDIR,sprintf('%s-downsample_round%.03i_colorcalcs.mat',params.FILE_BASENAME,rnd_indx));
    if ~exist(filename_colorShifts,'file')
        fprintf('Skipping missing shifts file %s \n',filename_colorShifts);
        continue;
    end    
    
    %Create the symlink of chan1 to the new directory
    chan1_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch00.tif',params.FILE_BASENAME,rnd_indx));
    command = sprintf('ln -s %s %s',chan1_inname,chan1_outname);
    system(command)
    fprintf('Created symlink %s \n',chan1_outname);
    
    chan2 = load3DTif_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch01.tif',params.FILE_BASENAME,rnd_indx)));
    chan2_shift = imtranslate3D(chan2,round(chan2_offsets*params.DOWNSAMPLE_RATE));
    chan2_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch01SHIFT.tif',params.FILE_BASENAME,rnd_indx));
    save3DTif_uint16(chan2_shift,chan2_outname);
    
    chan3 = load3DTif_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch02.tif',params.FILE_BASENAME,rnd_indx)));
    chan3_shift = imtranslate3D(chan3,round(chan3_offsets*params.DOWNSAMPLE_RATE));
    chan3_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch02SHIFT.tif',params.FILE_BASENAME,rnd_indx));
    save3DTif_uint16(chan3_shift,chan3_outname);
    
    chan4 = load3DTif_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch03.tif',params.FILE_BASENAME,rnd_indx)));
    chan4_shift = imtranslate3D(chan4,real(round(chan4_offsets*params.DOWNSAMPLE_RATE)));
    chan4_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch03SHIFT.tif',params.FILE_BASENAME,rnd_indx));
    save3DTif_uint16(chan4_shift,chan4_outname);
    
    
end
