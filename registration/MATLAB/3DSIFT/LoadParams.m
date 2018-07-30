%global TwoPeak_Flag IndexSize Display_flag Tessellation_flag Tessellation_levels nFaces Smooth_Flag Smooth_Var;
%global IgnoreGradSign IndexSigma MagFactor UseHistogramOri OriHistThresh OriSigma;

sift_params.TwoPeak_Flag = false;  % Allow 3DSIFT to throw out points, Default: true
sift_params.TwoPeak_Thresh = .9;  % Allow 3DSIFT to throw out points, Default: true
sift_params.IndexSize = 2;  % Min: 1  Default: 2 
sift_params.xyScale = 1;
sift_params.tScale = 1;

sift_params.Display_flag = 0;  % Display a sphere which can be rotated (Rotate 3D button) to view gradient directions
sift_params.Tessellation_flag = 1;  % Keep this as 1
sift_params.Tessellation_levels = 1;  % Min: zero  Default: 1
sift_params.Tessel_thresh = 3; % take the top nearest tesselations in GetGradOri / PlaceInIndex
sift_params.nFaces = 20 * ( 4 ^ sift_params.Tessellation_levels );  % Number of faces in the tessellation, not a parameter
sift_params.descriptor_len = sift_params.IndexSize *...
    sift_params.IndexSize * sift_params.IndexSize * sift_params.nFaces;

sift_params.Smooth_Flag = true;  % Adds gradient data to surrounding bins in final histogram
sift_params.Smooth_Var = 20;  % Determines amount of smoothing, Default: 20
sift_params.MaxIndexVal = 0.2;

% The rest of the variables are not modified often, but are included here
% for completeness sake
sift_params.IgnoreGradSign = 0;
sift_params.IndexSigma = 5.0;
sift_params.SigmaScaled = sift_params.IndexSigma * 0.5 * sift_params.IndexSize;
sift_params.MagFactor = 3; 

sift_params.UseHistogramOri = 1;
sift_params.OriHistThresh = 0.8;

if (sift_params.UseHistogramOri)
    sift_params.OriSigma = 1.5;
else
    sift_params.OriSigma = 1.0;
end
