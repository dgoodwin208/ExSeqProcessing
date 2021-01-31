bigparams.EXPERIMENT_NAME = 'htapp364';
bigparams.EXPERIMENT_FOLDERROOT = '/mp/nas2/DG/HTAPP_20200921/HTAPP_364/processing/';
bigparams.REGISTRATION_WORKINGDIR = fullfile(bigparams.EXPERIMENT_FOLDERROOT,'bigReg');
bigparams.EXPERIMENT_TILESIZE = [16,5];
bigparams.NUMTILES = prod(bigparams.EXPERIMENT_TILESIZE);
bigparams.NUMROUNDS = 6;
bigparams.DOWNSAMPLE_RATE = 4;
bigparams.REFERENCE_ROUND = 2; %TEMP: make sure this is the same as params.

bigparams.FOVS_TO_IGNORE = []; %[0, 1, 15:20,55:57,94:95,106:109];

bigparams.OUTPUTDIR = '/mp/nas2/DG/HTAPP_20200921/HTAPP_364';
bigparams.REG_CHANNEL = 'summedNorm';
bigparams.TILE_OVERLAP= .02;

bigparams.IMG_SIZE_XY = [2048, 2048];

bigparams.CHANNELS = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT','summedNorm'};
bigparams.IMAGE_EXT = 'h5';


%Create the spatial map of the tiling snake pattern we want to use
%A map of all the tile indices laid out in a 2D grid
bigparams.TILE_MAP= [];
for n = 1:bigparams.NUMTILES
    [row, col] = calculateRowColForSnakeTilePattern(n-1,bigparams.EXPERIMENT_TILESIZE(1));
    bigparams.TILE_MAP(row+1,col+1) = n-1;
end

%Critically, we must adjust the map so we can create a grid of images all
%inside the same coordinate frame as the dragonfly camera! 
bigparams.TILE_MAP = fliplr(transpose(bigparams.TILE_MAP));

