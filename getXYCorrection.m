function [ corr_y,corr_x ] = getXYCorrection( d1_max,d2_max )
%GETXYCORRECTION Summary of this function goes here
%   Detailed explanation goes here

WINSIZE = 20;

c = normxcorr2(d1_max,d2_max);

 c_sub = c(size(d1_max,1)-WINSIZE:size(d1_max,1)+WINSIZE,...
     size(d1_max,2)-WINSIZE:size(d1_max,2)+WINSIZE);

[zmax,imax,zmin,imin]= extrema2(c_sub);
[y, x] = ind2sub(size(c_sub),imax);

%Get the correction to warp d2 into d1
corr_y = (WINSIZE+1)-y(1);
corr_x = (WINSIZE+1)-x(1);




end

