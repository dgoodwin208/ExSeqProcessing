function [offsets3D] = colorcorrection_3D_from_file(roundnum)

loadParameters;

%Written for the first splintr barcode dataset from Oz
% FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = 'exseqautoframe7';
FILEROOT_NAME_INTERP = 'exseqautoframe7';
DIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampusOrig/1_original/';
DIRECTOR_SAVEDRESULTS = '/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170725snapshot/1_deconvolution/';

% NUM_ROUNDS = params.NUM_ROUNDS;
offsets3D = [6,6,5]; %X,Y,Z offsets for calcuating the difference
% BEAD_ZSTART = 120;
Z_upsample = 1.0; %0.5/.17;

opts_tiff.append = false;
opts_tiff.big = false;

%parfor roundnum = 1:NUM_ROUNDS
fprintf('Starting processing of round %i\n',roundnum);
%Load all channels, normalize them, calculate the cross corr of 
%channels 1-3 vs 4
tic; 

disp('load file 1');
chan1 = double(read_file(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch00.tif',FILEROOT_NAME_INTERP,roundnum))));

disp('load file 2');
chan2 = double(read_file(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01.tif',FILEROOT_NAME_INTERP,roundnum))));

disp('load file 3');
chan3 = double(read_file(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02.tif',FILEROOT_NAME_INTERP,roundnum))));

disp('load file 4');
chan4 = double(read_file(fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03.tif',FILEROOT_NAME_INTERP,roundnum))));

%The file contains: 'xcorr_scores3to1','xcorr_scores2to1','xcorr_scores4to1'
load(fullfile(DIRECTOR_SAVEDRESULTS,sprintf('%s_round%.03i_colorcalcs.mat',FILEROOT_NAME,roundnum)));


%Warp Chan4
mval = max(xcorr_scores4to1(:));
idx = find(mval==xcorr_scores4to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores4to1),idx);
chan4_offsets = [x_max,y_max,z_max] - (offsets3D+1);
chan4_offsets(3) = round(chan4_offsets(3)*Z_upsample);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,4,chan4_offsets(1),chan4_offsets(2),chan4_offsets(3));
 
disp('translate 4');
chan4_shift = imtranslate3D(chan4,chan4_offsets);

disp('save file 4');
saveastiff(uint16(chan4_shift),fullfile(DIRECTORY,sprintf('%s_round%.03i_ch03SHIFT.tif',FILEROOT_NAME_INTERP,roundnum)),opts_tiff);


%Warp Chan2
mval = max(xcorr_scores2to1(:));
idx = find(mval==xcorr_scores2to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores2to1),idx);
chan2_offsets = [x_max,y_max,z_max] - (offsets3D+1);
chan2_offsets(3) = round(chan2_offsets(3)*Z_upsample);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum,2,chan2_offsets(1),chan2_offsets(2),chan2_offsets(3));
disp('translate 2');
chan2_shift = imtranslate3D(chan2,chan2_offsets);

disp('save file 2');
saveastiff(uint16(chan2_shift),fullfile(DIRECTORY,sprintf('%s_round%.03i_ch01SHIFT.tif',FILEROOT_NAME_INTERP,roundnum)),opts_tiff);


%Warp Chan3
mval = max(xcorr_scores3to1(:));
idx = find(mval==xcorr_scores3to1(:));
[x_max,y_max,z_max] = ind2sub(size(xcorr_scores3to1),idx);
chan3_offsets = [x_max,y_max,z_max] - (offsets3D+1);
chan3_offsets(3) = round(chan3_offsets(3)*Z_upsample);
fprintf('Round %i: Offsets for chan%i: %i %i %i\n',roundnum, 3,chan3_offsets(1),chan3_offsets(2),chan3_offsets(3));

disp('translate 3');
chan3_shift = imtranslate3D(chan3,chan3_offsets);

disp('save file 3');
saveastiff(uint16(chan3_shift),fullfile(DIRECTORY,sprintf('%s_round%.03i_ch02SHIFT.tif',FILEROOT_NAME_INTERP,roundnum)),opts_tiff);


toc

end
