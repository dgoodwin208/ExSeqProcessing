bigparams.EXPERIMENT_NAME = 'htapp514';
bigparams.EXPERIMENT_FOLDERROOT = '/mp/nas2/DG/HTAPP_20200921/HTAPP_514/processing/';
bigparams.EXPERIMENT_TILESIZE = [19,6];
bigparams.NUMTILES = prod(bigparams.EXPERIMENT_TILESIZE);
bigparams.NUMROUNDS = 7;
bigparams.DOWNSAMPLE_RATE = 4;

bigparams.FOVS_TO_IGNORE = [0, 1, 15:20,55:57,94:95,106:109];

bigparams.OUTPUTDIR = '/mp/nas2/DG/HTAPP_20200921/HTAPP_514';
bigparams.REG_CHANNEL = 'summedNorm';
bigparams.TILE_OVERLAP= .02;

bigparams.IMG_SIZE_XY = [2048, 2048];
bigparams.IMG_RES_XY = .1625; % in um, Taken from the .ims properties read via FIJI
bigparams.IMG_RES_Z = 0.3974;



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

