function examineColorCorrection(roundnum)
%Written for the first splintr barcode dataset from Oz

FILEROOT_NAME = 'exseqautoframe7';
ROUND1DIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampus_rename/';
DIRECTORY = '/mp/nas0/kajita/062917_ExSeqAutoSeqHippo_allrounds_mod';
OUTPUT_DIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170809ColorCorrComparison';
offsets3D = [6,6,10]; %X,Y,Z offsets for calcuating the difference
BEAD_ZSTART = 120;
XYOFFSET = 470;

%parfor roundnum = 1:NUM_ROUNDS
fprintf('Starting processing of round %i\n',roundnum);
%Load all channels, normalize them, calculate the cross corr of 
%channels 1-3 vs 4
chan1 = load3DTif(fullfile(ROUND1DIRECTORY,sprintf('%s_round%.03i_ch00.tif',FILEROOT_NAME,roundnum)));
chan2 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01SHIFT.tif',FILEROOT_NAME,roundnum)));
chan3 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02SHIFT.tif',FILEROOT_NAME,roundnum)));
chan4 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));

chan1_beads = chan1(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);
chan2_beads = chan2(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);
chan3_beads = chan3(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);
chan4_beads = chan4(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);

save3DTif(chan1_beads,fullfile(OUTPUT_DIRECTORY,sprintf('BEADS%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));
save3DTif(chan2_beads,fullfile(OUTPUT_DIRECTORY,sprintf('BEADS%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));
save3DTif(chan3_beads,fullfile(OUTPUT_DIRECTORY,sprintf('BEADS%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));
save3DTif(chan4_beads,fullfile(OUTPUT_DIRECTORY,sprintf('BEADS%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));

end