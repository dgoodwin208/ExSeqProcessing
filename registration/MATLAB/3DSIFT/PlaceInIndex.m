function index = PlaceInIndex(index, mag, vect, i, j, s, fv, sift_params)

%Find the nearest tesselation face indices
corr_array = fv.centers * vect';

[yy ix] = sort(corr_array,'descend');

if (sift_params.Smooth_Flag == 0)
    index(i,j,s,ix(1)) = index(i,j,s,ix(1)) + mag;
elseif (sift_params.Smooth_Flag == 1)
    tmpsum = sum(yy(1:3).^sift_params.Smooth_Var);
    %Add to the three nearest tesselation faces
    for ii=1:3
        index(i,j,s,ix(ii)) = index(i,j,s,ix(ii)) + ( mag * ( yy(ii) .^ sift_params.Smooth_Var ) / tmpsum );
    end
end
