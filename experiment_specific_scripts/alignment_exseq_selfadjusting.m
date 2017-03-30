%Written for the first splintr barcode dataset from Oz
FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = 'exseqauto';

for roundnum = 1:30
    
    %Load all channels, normalize them, calculate the cross corr of 
    %channels 1-3 vs 4
    chan1 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch00fixed.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    chan2 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch01fixed.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    chan3 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch02fixed.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    chan4 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch03fixed.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    
    data_cols(:,1) = reshape(chan1,[],1);
    data_cols(:,2) = reshape(chan2,[],1);
    data_cols(:,3) = reshape(chan3,[],1);
    data_cols(:,4) = reshape(chan4,[],1);

    %     %Normalize the data
    data_cols_norm = quantilenorm(data_cols);

    % reshape the normed results back into 3d images
    chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
    chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
    chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
    chan4_norm = reshape(data_cols_norm(:,4),size(chan4));

    fixed_chans_norm = (chan1_norm + chan2_norm + chan3_norm)/3;
    max_fixed = max(fixed_chans_norm,[],3);
    max_moving = max(chan4_norm,[],3);
    [corr_y,corr_x] = getXYCorrection(max_fixed,max_moving);
    
    fprintf('Round%i has a corr_y=%i and corr_x=%i\n',roundnum,corr_y,corr_x);
    
%     %These numbers were calculated using the beads dataset from Shahar 
%     %given to Dan in January 2017
%     corr_y = 2;
%     corr_x = 4;
%     
%     %make the corrected Chan4 dataset
%     chan4_corr = zeros(size(chan4));
%     %Loop over every z index of the stack
%     for z = 1:size(chan4,3)
%         chan4_corr(:,:,z) = imtranslate(squeeze(chan4(:,:,z)),[corr_x,corr_y]);
%     end
%     
%     %Save the fourth channel as a corrected
%     save3DTif(chan4_corr,sprintf('/om/project/boyden/%s/input/%s_round%i_ch03corr.tif',FOLDER_NAME,FILEROOT_NAME, roundnum));
    
end
