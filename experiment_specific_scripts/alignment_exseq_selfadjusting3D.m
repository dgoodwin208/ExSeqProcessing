%Written for the first splintr barcode dataset from Oz
% FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = 'PrimerNPCCCframe7';

DIRECTORY = '/mp/nas0/ExSeq';

offsets3D = [6,6,5]; %X,Y,Z offsets for calcuating the difference
roundnum = 1;
BEAD_ZSTART = 120;

%Load all channels, normalize them, calculate the cross corr of 
%channels 1-3 vs 4
chan1 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch00.tif',FILEROOT_NAME,roundnum)));
chan2 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch01.tif',FILEROOT_NAME,roundnum)));
chan3 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch02.tif',FILEROOT_NAME,roundnum)));
chan4 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch03.tif',FILEROOT_NAME,roundnum)));
    
chan1_beads = chan1(:,:,BEAD_ZSTART:end);
chan2_beads = chan2(:,:,BEAD_ZSTART:end);
chan4_beads = chan4(:,:,BEAD_ZSTART:end);

xcorr_scores4to1 = crossCorr3D(chan1_beads,chan4_beads,offsets3D);  
save('sample_xcorr_scores.mat','xcorr_scores');

mval = max(xcorr_scores4to1(:));
idx = find(mval==xcorr_scores4to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores4to1),idx);
chan4_offsets = [x_max,y_max,z_max] - (offsets3D+1);

chan4_shift = imtranslate3D(chan4,chan4_offsets);
save3DTif(chan4_shift,fullfile(DIRECTORY,sprintf('%s_round%i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));

chan4_shift_beads = chan4_shift(:,:,BEAD_ZSTART:end);
xcorr_scores2to1 = crossCorr3D(chan1_beads+chan4_shift_beads,chan2_beads,offsets3D);
mval = max(xcorr_scores2to1(:));
idx = find(mval==xcorr_scores2to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores2to1),idx);
chan2_offsets = [x_max,y_max,z_max] - (offsets3D+1);
chan2_shift = imtranslate3D(chan2,chan2_offsets);
save3DTif(chan2_shift,fullfile(DIRECTORY,sprintf('%s_round%i_ch01SHIFT.tif',FILEROOT_NAME,roundnum)));


data_cols(:,1) = reshape(chan1,[],1);
data_cols(:,2) = reshape(chan2_shift,[],1);
data_cols(:,3) = reshape(chan3,[],1);
data_cols(:,4) = reshape(chan4_shift,[],1);

%     %Normalize the data
data_cols_norm = quantilenorm(data_cols);

% reshape the normed results back into 3d images
chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
chan4_norm = reshape(data_cols_norm(:,4),size(chan4));

fixed_chans_norm = (chan1_norm + chan2_norm + chan4_norm)/3;

xcorr_scores3to1 = crossCorr3D(fixed_chans_norm,chan3_norm,offsets3D);
mval = max(xcorr_scores3to1(:));
idx = find(mval==xcorr_scores3to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores3to1),idx);
chan3_offsets = [x_max,y_max,z_max] - (offsets3D+1);
chan3_shift = imtranslate3D(chan3,chan3_offsets);
save3DTif(chan3_shift,fullfile(DIRECTORY,sprintf('%s_round%i_ch02SHIFT.tif',FILEROOT_NAME,roundnum)));

save('processing_results.mat','xcorr_scores3to1','xcorr_scores2to1','xcorr_scores');
