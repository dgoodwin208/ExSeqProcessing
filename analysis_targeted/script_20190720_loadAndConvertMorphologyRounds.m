% Script that should contain all necessary metadata for takign the chM
% rounds and stitching them together complete
% Written by Dan Goodwin, dgoodwin@mit.edu July 20 2019

%% Create the spatial map of the tiling snake pattern
% Numbers are from metadata and examining the raw data folders
numTiles = 180;
numRows = 10;

%A map of all the tile indices laid out in a 2D grid 
tileMap_indices = zeros(10,18);
%The top left xy position of a tile in microns
tileMap_spatialOffsets = cell(10,18); 

imgSizeXY = [2048, 2048];
imgResXY = .1625; % in uM, Taken from the .ims properties read via FIJI
imgResZ = 0.3974;
for n = 1:numTiles
    [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
    tileMap_indices(row+1,col+1) = n-1;
    tileMap_spatialOffsets{row+1,col+1} = ...
        [row*imgSizeXY(1)*imgResXY, col*imgSizeXY(2)*imgResXY];     
end

