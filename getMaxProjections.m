% Quick script to get a maxProjection of all images to show
% the difference between pre/post registration

FOLDER_NAME = 'ExSeqSlice';
FILEROOT_NAME = 'sa0916slicedncv';
for roundnum = 1:12
        pre_reg_summed = load3DTif(sprintf('/om/project/boyden/%s/input/%s_round%i_summedNorm.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
        post_reg_summed = load3DTif(sprintf('/om/project/boyden/%s/output/FULLTPS%s_round%i_summedNorm.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
        
        max_pre_reg_summed = max(pre_reg_summed,[],3);
        max_post_reg_summed = max(post_reg_summed,[],3);
        
        save3DTif(max_pre_reg_summed,sprintf('/om/project/boyden/%s/output/MAXPROJ%s_round%i_summedNorm.tif',FOLDER_NAME,FILEROOT_NAME, roundnum));
        save3DTif(max_post_reg_summed, sprintf('/om/project/boyden/%s/output/MAXPROJFULLTPS%s_round%i_summedNorm.tif',FOLDER_NAME,FILEROOT_NAME,roundnum));
end
