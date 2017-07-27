function [offsets3D] = colorcorrection_3D(roundnum)
%Written for the first splintr barcode dataset from Oz
% FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = 'exseqautoframe7i';
DIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampus_rename/';
NUM_ROUNDS = 20;
offsets3D = [6,6,10]; %X,Y,Z offsets for calcuating the difference
BEAD_ZSTART = 120*3;
XYOFFSET = 470;

%parfor roundnum = 1:NUM_ROUNDS
fprintf('Starting processing of round %i\n',roundnum);
%Load all channels, normalize them, calculate the cross corr of 
%channels 1-3 vs 4
chan1 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch00.tif',FILEROOT_NAME,roundnum)));
chan2 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01.tif',FILEROOT_NAME,roundnum)));
chan3 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02.tif',FILEROOT_NAME,roundnum)));
chan4 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03.tif',FILEROOT_NAME,roundnum)));
    
chan1_beads = chan1(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);
chan2_beads = chan2(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);
chan3_beads = chan3(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);
chan4_beads = chan4(XYOFFSET:end,XYOFFSET:end,BEAD_ZSTART:end);

xcorr_scores4to1 = crossCorr3D(chan1_beads,chan4_beads,offsets3D);  
mval = max(xcorr_scores4to1(:));
idx = find(mval==xcorr_scores4to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores4to1),idx);
chan4_offsets = [x_max,y_max,z_max] - (offsets3D+1);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,4,chan4_offsets(1),chan4_offsets(2),chan4_offsets(3));
chan4_shift = imtranslate3D(chan4,chan4_offsets);
save3DTif(chan4_shift,fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));

%chan4_shift_beads = chan4_shift(:,:,BEAD_ZSTART:end);
xcorr_scores2to1 = crossCorr3D(chan1_beads,chan2_beads,offsets3D);
mval = max(xcorr_scores2to1(:));
idx = find(mval==xcorr_scores2to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores2to1),idx);
chan2_offsets = [x_max,y_max,z_max] - (offsets3D+1);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,2,chan2_offsets(1),chan2_offsets(2),chan2_offsets(3));
chan2_shift = imtranslate3D(chan2,chan2_offsets);
save3DTif(chan2_shift,fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01SHIFT.tif',FILEROOT_NAME,roundnum)));

xcorr_scores3to1 = crossCorr3D(chan1_beads,chan3_beads,offsets3D);
mval = max(xcorr_scores3to1(:));
idx = find(mval==xcorr_scores3to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores3to1),idx);
chan3_offsets = [x_max,y_max,z_max] - (offsets3D+1);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum, 3,chan3_offsets(1),chan3_offsets(2),chan3_offsets(3));

chan3_shift = imtranslate3D(chan3,chan3_offsets);
save3DTif(chan3_shift,fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02SHIFT.tif',FILEROOT_NAME,roundnum)));

save(fullfile(DIRECTORY,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,roundnum)),'xcorr_scores3to1','xcorr_scores2to1','xcorr_scores4to1');
end
