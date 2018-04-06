function [index precomp_grads] = AddSamplePrecomp(index, pix, distsq, r, c, s, i, j, s, fv, sift_params, precomp_grads)
% r, c, s is the pixel index (x, y, z dimensions respect.) in the image within the radius of the 
% keypoint before clamped
% For each pixel, take a neighborhhod of xyradius and tiradius,
% bin it down to the sift_params.IndexSize dimensions
% thus, i_indx, j_indx, s_indx represent the binned index within the radius of the keypoint

%FIXME put this in the sift_params file
sigma = sift_params.SigmaScaled;
weight = exp(-double(distsq / (2.0 * sigma * sigma)));

% Check if computed the gradient previously before
key = sub2ind(sift_params.pix_size, r,c,s);
if isKey(precomp_grads, key)
    cell_val = precomp_grads(key);
    %precomp_grads(key) = 
else

    rows = sift_params.pix_size(1);
    cols = sift_params.pix_size(2);
    slices = sift_params.pix_size(3);

    if (c == 1)
        xgrad = 2.0 * (double(pix(r,c+1,s)) - double(pix(r,c,s)));
    elseif (c == cols)
        xgrad = 2.0 * (double(pix(r,c,s)) - double(pix(r,c-1,s)));
    else
        xgrad = double(pix(r,c+1,s)) - double(pix(r,c-1,s));
    end
    if (r == 1)
        ygrad = 2.0 * (double(pix(r,c,s)) - double(pix(r+1,c,s)));
    elseif (r == rows)
        ygrad = 2.0 * (double(pix(r-1,c,s)) - double(pix(r,c,s)));
    else
        ygrad = double(pix(r-1,c,s)) - double(pix(r+1,c,s));
    end
    if (s == 1)
        zgrad = 2.0 * (double(pix(r,c,s+1)) - double(pix(r,c,s)));
    elseif (s == slices)
        zgrad = 2.0 * (double(pix(r,c,s)) - double(pix(r,c,s-1)));
    else
        zgrad = double(pix(r,c,s+1)) - double(pix(r,c,s-1));
    end

    xgrad = double(xgrad);
    ygrad = double(ygrad);
    zgrad = double(zgrad);

    mag = sqrt(xgrad * xgrad + ygrad * ygrad + zgrad * zgrad);

    if mag ~=0
        vect = [xgrad ygrad zgrad] ./ mag;
    else
        vect = [1 0 0];
    end


    %mag = weight * mag; %scale magnitude by gaussian 

    %Find the nearest tesselation face indices
    corr_array = fv.centers * vect';

    [yy ix] = sort(corr_array,'descend');

    cell_val = {1, mag_vect};
    precomp_grads(key) = cell_val;
end

% Add to index
if (sift_params.Smooth_Flag == 0)
    index(i,j,s,ix(1)) = index(i,j,s,ix(1)) + mag;
elseif (sift_params.Smooth_Flag == 1)
    tmpsum = sum(yy(1:3).^sift_params.Smooth_Var);
    %Add to the three nearest tesselation faces
    for ii=1:3
        index(i,j,s,ix(ii)) = index(i,j,s,ix(ii)) + ( mag * ( yy(ii) .^ sift_params.Smooth_Var ) / tmpsum );
    end
end

end
 
