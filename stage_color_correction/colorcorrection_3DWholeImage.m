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

num_channels = params.NUM_CHANNELS;
chan_strs = params.CHAN_STRS;
if roundnum == params.MORPHOLOGY_ROUND
    num_channels = num_channels + 1;
    chan_strs{num_channels} = params.MORPHOLOGY_CHAN_STR;
end

imgR = load3DImage_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_%s.%s',FILEROOT_NAME,roundnum,chan_strs{1},params.IMAGE_EXT)));

chan_offsets = zeros(num_channels,3);
for c = 2:num_channels
	imgM = load3DImage_uint16(fullfile(DIRECTORY,sprintf('%s_round%.03i_%s.%s',FILEROOT_NAME,roundnum,chan_strs{c},params.IMAGE_EXT)));

	offsets = phaseOnlyCorrelation(imgR,imgM,[20 20 20]);

	imgM_shift = imtranslate3D(imgM,offsets);
	chan_offsets(c,:) = offsets;
 
	save3DImage_uint16(imgM_shift,fullfile(OUTPUTDIR,sprintf('%s_round%0.3i_%sSHIFT.%s',FILEROOT_NAME,roundnum,chan_strs{c},params.IMAGE_EXT)));
end
	%TEMP: print the array of channel shifts:
 	chan_offsets	
	oldMethod = false;
	save(fullfile(OUTPUTDIR,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,roundnum)),...
    		'chan_offsets','oldMethod');
end
