function colorcorrection_3DWholeImage(roundnum)

loadParameters;

if params.DO_DOWNSAMPLE
    FILEROOT_NAME = sprintf('%s-downsample',params.FILE_BASENAME);
else
    FILEROOT_NAME = sprintf('%s',params.FILE_BASENAME);
end

DIRECTORY = params.deconvolutionImagesDir;
OUTPUTDIR = params.colorCorrectionImagesDir;


if exist(fullfile(OUTPUTDIR,sprintf('%s_round%.03i_ch02SHIFT.%s',FILEROOT_NAME,roundnum,params.IMAGE_EXT)),'file' );
    fprintf('SEES ch02SHIFT file in the output directory. Skipping\n');
    return
end

fprintf('Starting processing of round %i\n',roundnum);
%Load all channels, normalize them, calculate the cross corr of 
%channels 1-3 vs 4

imgR = load3DImage_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch0%i.%s',FILEROOT_NAME,roundnum,1-1,params.IMAGE_EXT)));


chan_offsets = zeros(4,3);
for c = 2:params.NUM_CHANNELS
	imgM = load3DImage_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch0%i.%s',FILEROOT_NAME,roundnum,c-1,params.IMAGE_EXT)));

	offsets = phaseOnlyCorrelation(imgR,imgM,[20 20 20]);

	imgM_shift = imtranslate3D(imgM,offsets);
	chan_offsets(c,:) = offsets;
 
	save3DImage_uint16(imgM_shift,fullfile(OUTPUTDIR,sprintf('%s_round%0.3i_ch%0.2iSHIFT.%s',FILEROOT_NAME,roundnum,c-1,params.IMAGE_EXT)));
end
	%TEMP: print the array of channel shifts:
 	chan_offsets	
	oldMethod = false;
	save(fullfile(OUTPUTDIR,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,roundnum)),...
    		'chan_offsets','oldMethod');
end
