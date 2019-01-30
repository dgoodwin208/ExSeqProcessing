%Written for the first splintr barcode dataset from Oz
FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = 'exseqauto';

for roundnum = 1:30
    %Load three of the channels, ignoring ch02
    chan4 = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_ch03.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
    
    %These numbers were calculated using the beads dataset from Shahar 
    %given to Dan in January 2017
    corr_y = 2;
    corr_x = 4;
    
    %make the corrected Chan4 dataset
    chan4_corr = zeros(size(chan4));
    %Loop over every z index of the stack
    for z = 1:size(chan4,3)
        chan4_corr(:,:,z) = imtranslate(squeeze(chan4(:,:,z)),[corr_x,corr_y]);
    end
    
    %Save the fourth channel as a corrected
    save3DTif(chan4_corr,sprintf('/om/project/boyden/%s/input/%s_round%i_ch03corr.tif',FOLDER_NAME,FILEROOT_NAME, roundnum));
    
end
