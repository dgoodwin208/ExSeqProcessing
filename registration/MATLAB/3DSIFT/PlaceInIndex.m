function index = PlaceInIndex(index, mag, i, j, s, yy, ix, sift_params)

if (sift_params.Smooth_Flag == 0)
    index(i,j,s,ix(1)) = index(i,j,s,ix(1)) + mag;
elseif (sift_params.Smooth_Flag == 1)
    tmpsum = sum(yy(1:sift_params.Tessel_thresh).^sift_params.Smooth_Var);
    %Add to the three nearest tesselation faces
    for ii=1:sift_params.Tessel_thresh
        bin_index = sub2ind(size(index), i, j, s, ix(ii));
        %fprintf('ML i%d j%d k%d ix[ii]%d bin_index%d yy[ii]%f, index+=%.3f\n', i, j, s, ...
                %ix(ii) - 1, bin_index - 1, yy(ii), ...
                %mag * (yy(ii) .^ sift_params.Smooth_Var ) / tmpsum);
        index(i,j,s,ix(ii)) = index(i,j,s,ix(ii)) + ( mag * ( yy(ii) .^ sift_params.Smooth_Var ) / tmpsum );
    end
end
