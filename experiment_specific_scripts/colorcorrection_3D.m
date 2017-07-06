function [offsets3D] = colorcorrection_3D(roundnum)

loadParameters;

%Written for the first splintr barcode dataset from Oz
% FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = params.FILE_BASENAME;
DIRECTORY = params.colorCorrectionImagesDir;
NUM_ROUNDS = params.NUM_ROUNDS;
offsets3D = [6,6,5]; %X,Y,Z offsets for calcuating the difference
BEAD_ZSTART = 120;

%parfor roundnum = 1:NUM_ROUNDS
fprintf('Starting processing of round %i\n',roundnum);
%Load all channels, normalize them, calculate the cross corr of 
%channels 1-3 vs 4
tic; disp('load file 1');
chan1 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch00.tif',FILEROOT_NAME,roundnum)));
toc
tic; disp('load file 2');
chan2 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01.tif',FILEROOT_NAME,roundnum)));
toc
tic; disp('load file 3');
chan3 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02.tif',FILEROOT_NAME,roundnum)));
toc
tic; disp('load file 4');
chan4 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03.tif',FILEROOT_NAME,roundnum)));
toc
    
chan1_beads = chan1(:,:,BEAD_ZSTART:end);
chan2_beads = chan2(:,:,BEAD_ZSTART:end);
chan4_beads = chan4(:,:,BEAD_ZSTART:end);

tic; disp('crossCorr3D 4');
xcorr_scores4to1 = crossCorr3D(chan1_beads,chan4_beads,offsets3D);  
toc

mval = max(xcorr_scores4to1(:));
idx = find(mval==xcorr_scores4to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores4to1),idx);
chan4_offsets = [x_max,y_max,z_max] - (offsets3D+1);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,4,chan4_offsets(1),chan4_offsets(2),chan4_offsets(3));
tic; disp('translate 4');
chan4_shift = imtranslate3D(chan4,chan4_offsets);
toc
tic; disp('save file 4');
save3DTif(chan4_shift,fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME,roundnum)));
toc

chan4_shift_beads = chan4_shift(:,:,BEAD_ZSTART:end);
tic; disp('crossCorr3D 2');
xcorr_scores2to1 = crossCorr3D(chan1_beads+chan4_shift_beads,chan2_beads,offsets3D);
toc
mval = max(xcorr_scores2to1(:));
idx = find(mval==xcorr_scores2to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores2to1),idx);
chan2_offsets = [x_max,y_max,z_max] - (offsets3D+1);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,2,chan2_offsets(1),chan2_offsets(2),chan2_offsets(3));
tic; disp('translate 2');
chan2_shift = imtranslate3D(chan2,chan2_offsets);
toc
tic; disp('save file 2');
save3DTif(chan2_shift,fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01SHIFT.tif',FILEROOT_NAME,roundnum)));
toc

data_cols = zeros(length(reshape(chan1,[],1)),4);
data_cols(:,1) = reshape(chan1,[],1);
data_cols(:,2) = reshape(chan2_shift,[],1);
data_cols(:,3) = reshape(chan3,[],1);
data_cols(:,4) = reshape(chan4_shift,[],1);

%     %Normalize the data
tic; disp('quantilenorm');
data_cols_norm = quantilenorm(data_cols);
toc

% reshape the normed results back into 3d images
chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
chan4_norm = reshape(data_cols_norm(:,4),size(chan4));

fixed_chans_norm = (chan1_norm + chan2_norm + chan4_norm)/3;

%Clear up memory
clear data_cols data_cols_norm

tic; disp('crossCorr3D 3');
xcorr_scores3to1 = crossCorr3D(fixed_chans_norm,chan3_norm,offsets3D);
toc
mval = max(xcorr_scores3to1(:));
idx = find(mval==xcorr_scores3to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores3to1),idx);
chan3_offsets = [x_max,y_max,z_max] - (offsets3D+1);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum, 3,chan3_offsets(1),chan3_offsets(2),chan3_offsets(3));

tic; disp('translate 3');
chan3_shift = imtranslate3D(chan3,chan3_offsets);
toc
tic; disp('save file 3');
save3DTif(chan3_shift,fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02SHIFT.tif',FILEROOT_NAME,roundnum)));
toc

save(fullfile(DIRECTORY,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,roundnum)),'xcorr_scores3to1','xcorr_scores2to1','xcorr_scores4to1');
end
