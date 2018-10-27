%%

%load a token image so we get the size
img = load3DTif('/Users/Goody/Neuro/ExSeq/registration_analysis/20170813regtest/exseqautoframe7_round004_summedNorm_registered.tif');

load('/Users/Goody/Neuro/ExSeq/exseq20170524/exseqautoframe7_puncta_filtered.mat');
%load the puncta_allexp.mat file

%%
%Make a 3D image of the keypoints
keypoints_in_space = zeros(size(img));
SIZE=2;
for key_idx = 1:size(puncta_filtered,1)
    pos = puncta_filtered(key_idx,:);
    keypoints_in_space(pos(1)-SIZE:pos(1)+SIZE,...
        pos(2)-SIZE:pos(2)+SIZE,...
        pos(3)-SIZE:pos(3)+SIZE)=255;
end
save3DTif(keypoints_in_space,'filteredPuncta.tif');