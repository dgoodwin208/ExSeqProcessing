function colorcorrection_3D_from_file(roundnum)

loadParameters;

if params.DO_DOWNSAMPLE
    FILEROOT_NAME = sprintf('%s-downsample',params.FILE_BASENAME);
else
    FILEROOT_NAME = sprintf('%s',params.FILE_BASENAME);
end

DIRECTORY = params.deconvolutionImagesDir;
OUTPUTDIR = params.colorCorrectionImagesDir;

%Load the channel offsets
load(fullfile(OUTPUTDIR,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,roundnum)));


fprintf('Starting processing of round %i\n',roundnum);

disp('load file 2');
chan2 = load3DTif_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01.tif',FILEROOT_NAME,roundnum)));
disp('load file 3');
chan3 = load3DTif_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02.tif',FILEROOT_NAME,roundnum)));
disp('load file 4');
chan4 = load3DTif_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03.tif',FILEROOT_NAME,roundnum)));


fprintf('Round %i: Manual Offsets for chan%i: %i %i %i\n',roundnum,4,chan4_offsets(1),chan4_offsets(2),chan4_offsets(3));
chan4_shift = imtranslate3D(chan4,real(round(chan4_offsets)));
disp('save file 4');
save3DTif_uint16(chan4_shift,fullfile(OUTPUTDIR,sprintf('%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));

fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,2,chan2_offsets(1),chan2_offsets(2),chan2_offsets(3));
disp('translate 2');
chan2_shift = imtranslate3D(chan2,round(chan2_offsets));
disp('save file 2');
save3DTif_uint16(chan2_shift,fullfile(OUTPUTDIR,sprintf('%s_round%.03i_ch01SHIFT.tif',FILEROOT_NAME,roundnum)));

fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum, 3,chan3_offsets(1),chan3_offsets(2),chan3_offsets(3));
disp('translate 3');
chan3_shift = imtranslate3D(chan3,round(chan3_offsets));
disp('save file 3');
save3DTif_uint16(chan3_shift,fullfile(OUTPUTDIR,sprintf('%s_round%.03i_ch02SHIFT.tif',FILEROOT_NAME,roundnum)));

%And, if the color correction was done on downsampled data, now apply it to
%full res data
if params.DO_DOWNSAMPLE
   chan1_inname = fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch00.tif',params.FILE_BASENAME,roundnum));
    
    if ~exist(chan1_inname,'file')
        fprintf('Skipping missing round %i \n',roundnum);
        return;
    end
    
    chan4_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch03SHIFT.tif',params.FILE_BASENAME,roundnum));
    
    %Create the symlink of chan1 to the new directory
    chan1_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch00.tif',params.FILE_BASENAME,roundnum));
    command = sprintf('ln -s %s %s',chan1_inname,chan1_outname);
    system(command);
    fprintf('Created symlink %s \n',chan1_outname);
    
    chan2 = load3DTif_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch01.tif',params.FILE_BASENAME,roundnum)));
    chan2_shift = imtranslate3D(chan2,round(chan2_offsets*params.DOWNSAMPLE_RATE));
    chan2_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch01SHIFT.tif',params.FILE_BASENAME,roundnum));
    save3DTif_uint16(chan2_shift,chan2_outname);
    
    chan3 = load3DTif_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch02.tif',params.FILE_BASENAME,roundnum)));
    chan3_shift = imtranslate3D(chan3,round(chan3_offsets*params.DOWNSAMPLE_RATE));
    chan3_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch02SHIFT.tif',params.FILE_BASENAME,roundnum));
    save3DTif_uint16(chan3_shift,chan3_outname);
    
    chan4 = load3DTif_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch03.tif',params.FILE_BASENAME,roundnum)));
    chan4_shift = imtranslate3D(chan4,real(round(chan4_offsets*params.DOWNSAMPLE_RATE)));
    save3DTif_uint16(chan4_shift,chan4_outname);
     
end

end
