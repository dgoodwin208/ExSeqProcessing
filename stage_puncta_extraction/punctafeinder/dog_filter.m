function img_filtered = dog_filter(img)
%returns dog filtered img
%% DoG signal enhancement/filter
tic
FILTER_SIZE_XY = 6; % for [fxf] filter 
FILTER_SIZE_Z = 6; 
STDXY1 = 3;%4 
STDXY2 = 4;%5 % btw, spots are black not white if std1 > std2
STDZ1 = 3; 
STDZ2 = 4; 

% 3d DoG Gaussian filters
gauss1 = gauss_filter_fun_xyz(FILTER_SIZE_XY, FILTER_SIZE_Z, STDXY1, STDZ1);
gauss2 = gauss_filter_fun_xyz(FILTER_SIZE_XY, FILTER_SIZE_Z, STDXY2, STDZ2);
dgauss = gauss1 - gauss2;
img_filtered = imfilter(img, dgauss, 'symmetric', 'conv', 'same'); 
fprintf('dog\n');
toc