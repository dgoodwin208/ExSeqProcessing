bigparams.EXPERIMENT_NAME = 'htapp514';
bigparams.EXPERIMENT_FOLDERROOT = '/mp/nas2/DG/HTAPP_20200921/HTAPP_514/processing/';
bigparams.EXPERIMENT_TILESIZE = [19,6];
bigparams.NUMROUNDS = 7;
bigparams.DOWNSAMPLE_RATE = 4;

bigparams.FOVS_TO_IGNORE = [0, 1, 15:20,55:57,94:95,106:109];

bigparams.OUTPUTDIR = '/mp/nas2/DG/HTAPP_20200921/HTAPP_514';
bigparams.REG_CHANNEL = 'summedNorm';
bigparams.TILE_OVERLAP= .02;

bigparams.IMG_SIZE_XY = [2048, 2048];
bigparams.IMG_RES_XY = .1625; % in um, Taken from the .ims properties read via FIJI
bigparams.IMG_RES_Z = 0.3974;
