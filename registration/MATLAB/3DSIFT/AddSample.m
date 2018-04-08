function [index precomp_grads] = AddSample(index, pix, distsq, r, c, s, i_indx, j_indx, s_indx, fv, sift_params, precomp_grads)
% r, c, s is the pixel index (x, y, z dimensions respect.) in the image within the radius of the 
% keypoint before clamped
% For each pixel, take a neighborhhod of xyradius and tiradius,
% bin it down to the sift_params.IndexSize dimensions
% thus, i_indx, j_indx, s_indx represent the binned index within the radius of the keypoint

%FIXME put this in the sift_params file
sigma = sift_params.SigmaScaled;
weight = exp(-double(distsq / (2.0 * sigma * sigma)));

%[mag vect] from the immediately neighboring pixels
[mag vect precomp_grads, yy, ix] = GetGradOri_vector(pix,r,c,s, fv, sift_params, precomp_grads);
mag = weight * mag; %scale magnitude by gaussian 

index = PlaceInIndex(index, mag, vect, i_indx, j_indx, s_indx, yy, ix, sift_params);

end
 
