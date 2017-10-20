function filter = gauss_filter_fun(FILTER_SIZEXY, FILTER_SIZEZ, STDXY, STDZ, DIM)% filter size is scalar plz. gives (filter_size)^DIM filter
% Maryann Rui 08/2017
%% 2d and 3d gaussian kernel by scratch
if nargin < 5
    DIM = 3; % default 3D filter
end
switch DIM
%% manual 2d [same as fspecial('gaussian', FILTER_SIZE, STD)
    case 2
        if mod(FILTER_SIZEXY, 2) == 0 % even grid length
           % still centered around 0, with 1 increment [matches mlab!]
            grid = [-FILTER_SIZEXY/2+0.5:-0.5 0.5:FILTER_SIZEXY/2-0.5]; 
        else % odd grid length
            grid = -FILTER_SIZEXY/2:FILTER_SIZEXY/2; 
        end
        [X, Y] = meshgrid(grid, grid);
        % Create Gaussian Mask
        man_gauss2 = exp(-(X.^2 + Y.^2) / (2*STDXY^2));
        % Normalize so that total area (sum of all weights) is 1
        filter = man_gauss2/sum(man_gauss2(:));
%% manual 3d
    case 3
        if mod(FILTER_SIZEXY, 2) == 0 % even grid length
           % still centered around 0, with 1 increment [matches mlab!]
            gridxy = [-FILTER_SIZEXY/2+0.5:-0.5 0.5:FILTER_SIZEXY/2-0.5]; 
        else % odd grid length
            gridxy = -FILTER_SIZEXY/2:FILTER_SIZEXY/2; 
        end
        if mod(FILTER_SIZEZ, 2) == 0 % even
            gridz = [-FILTER_SIZEZ/2+0.5:-0.5 0.5:FILTER_SIZEZ/2-0.5]; 
        else % odd grid length
            gridz = -FILTER_SIZEZ/2:FILTER_SIZEZ/2; 
        end
        [X, Y, Z] = meshgrid(gridxy, gridxy, gridz);

        % Create Gaussian Mask
        man_gauss3 = exp(-(X.^2 + Y.^2) / (2*STDXY^2)  - Z.^2 / (2*STDZ^2));

        % Normalize so that total area (sum of all weights) is 1
        filter = man_gauss3/sum(man_gauss3(:));
end
end