%Written for the first splintr barcode dataset from Oz
FOLDER_NAME = 'Splintr';
FILEROOT_NAME = 'splintr1';

for roundnum = 1:3
    %Load three of the channels, ignoring ch02
    chan1 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch00.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    chan2 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch01.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    chan4 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch03.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    
    d1_max = max(chan1,[],3);
    d4_max = max(chan4,[],3);
    [ corr_y,corr_x ] = getXYCorrection( d1_max,d4_max );
    
    %make the corrected Chan4 dataset
    chan4_corr = zeros(size(chan4));
    %Loop over every z index of the stack
    for z = 1:size(chan4,3)
        chan4_corr(:,:,z) = imtranslate(squeeze(chan4(:,:,z)),[corr_x,corr_y]);
    end
    
    %Save the fourth channel as a corrected
    save3DTif(summed_norm,sprintf('/om/project/boyden/%s/input/%s_round%i_ch03corr.tif',FOLDER_NAME,FILEROOT_NAME, roundnum));
    
    %Note the sum of the channel
    summed = chan1+chan2+chan4_corr;
    
    data_cols(:,1) = reshape(chan1,[],1);
    data_cols(:,2) = reshape(chan2,[],1);
    data_cols(:,3) = reshape(chan4_corr,[],1);
    
    % Normalize the data
    data_cols_norm = quantilenorm(data_cols);
    
    % reshape the normed results back into 3d images
    chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
    chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
    chan4_norm = reshape(data_cols_norm(:,3),size(chan4_corr));
    
    
    summed_norm = chan1_norm+chan2_norm+chan4_norm;
    
    save3DTif(summed_norm,sprintf('/om/project/boyden/%s/input/%s_round%i_summedNorm.tif',FOLDER_NAME,FILEROOT_NAME, roundnum));
    
    save3DTif(chan1_norm, sprintf('/om/project/boyden/%s/input/%s_round%i_ch00Norm.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    save3DTif(chan2_norm, sprintf('/om/project/boyden/%s/input/%s_round%i_ch01Norm.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    save3DTif(chan4_norm, sprintf('/om/project/boyden/%s/input/%s_round%i_ch03corrNorm.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    
end

