function [keypoint reRun precomp_grads] = Create_Descriptor(pix, xyScale, tScale, x, y, z, sift_params, precomp_grads)
% Main function of 3DSIFT Program from http://www.cs.ucf.edu/~pscovann/
% 
% Inputs:
% pix - a 3 dimensional matrix of uint8
% xyScale and tScale - affects both the scale and the resolution, these are
% usually set to 1 and scaling is done before calling this function
% x, y, and z - the location of the center of the keypoint where a descriptor is requested
%
% Outputs:
% keypoint - the descriptor, varies in size depending on values in LoadParams.m
% reRun - a flag (0 or 1) which is set if the data at (x,y,z) is not
% descriptive enough for a good keypoint
%
% Example:
% See Demo.m

reRun = 0;

radius = int16(xyScale * 3.0);

fv = sphere_tri('ico',sift_params.Tessellation_levels,1);

if (sift_params.TwoPeak_Flag)

    [myhist, precomp_grads] = buildOriHists(x,y,z,radius,pix,fv,sift_params, precomp_grads);

    [yy ix] = sort(myhist,'descend');
    % Dom_Peak = ix(1);
    % Sec_Peak = ix(2);
        
if ((x == 1733) && (y == 22) && (z == 14))
    fprintf('mlab x%d y%d z%d ori_hist %.4f %.4f %.54f %.54f ori_hist_idx: %d %d %d %d eq:%d, diff:%.54f\n', ...
        x-1, y-1, z-1, yy(1), yy(2), yy(3), yy(4), ix(1) - 1, ix(2) - 1, ix(3) - 1, ix(4) - 1, yy(3) == yy(4), yy(3) - yy(4));
end


    if (dot(fv.centers(ix(1),:),fv.centers(ix(2),:)) > .9 && ...
        dot(fv.centers(ix(1),:),fv.centers(ix(3),:)) > .9)
        disp('MISS : Top 3 orientations within ~25 degree range : Returning with reRun flag set.');
        keypoint = 0;
        reRun = 1;
        return;
    end
end

[keypoint precomp_grads] = MakeKeypoint(pix, xyScale, tScale, x, y, z, sift_params, precomp_grads);

