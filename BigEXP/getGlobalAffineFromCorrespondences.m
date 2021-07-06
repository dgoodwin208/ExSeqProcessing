%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function description to go here
%%%%%%%%%%%%%%%%%%%%
function [affine_tform,keyM_total_transformed] = getGlobalAffineFromCorrespondences(keyM_total,keyF_total)




%First we do do a global affine transform on the data and keypoints before
%doing the fine-resolution non-rigid warp

%Because of the annoying switching between XY/YX conventions,
%we have to switch XY components for the affine calcs, then switch back
keyM_total_switch = keyM_total(:,[2 1 3]);
keyF_total_switch = keyF_total(:,[2 1 3]);

warning('off','all'); 
%Hardcoding that we're specifically looking for full affine (true), reather
%than a simplified translation only model
affine_tform = findAffineModel(keyM_total_switch, keyF_total_switch,true);
warning('on','all')

%note: affine_tform is now ready to be used in a line like:
%imgMoving_total_affine = imwarp(imgToWarp,affine3d(affine_tform'),'OutputView',rF);
%However, if we want to explore the warped keypoints, we have to toggle
%them back into their original format:
keyM_total_transformed = [keyM_total_switch, ones(size(keyM_total_switch,1),1)]*affine_tform';
keyM_total_transformed = keyM_total_transformed(:,1:3);
keyM_total_transformed = keyM_total_transformed(:,[2 1 3]);

end


