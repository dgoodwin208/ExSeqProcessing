function bigparams = loadBigExpParams(yamlfile)

%Use a YAML file to load the parameters for a large experiment
yamlspecs = ReadYaml(yamlfile);
%NOTE: The YAML files might have extra quotes so remove them

bigparams.EXPERIMENT_NAME = strrep(yamlspecs.basename,'''','');
bigparams.EXPERIMENT_FOLDERROOT = strrep(yamlspecs.base_dir,'''','');
bigparams.REGISTRATION_WORKINGDIR = fullfile(bigparams.EXPERIMENT_FOLDERROOT,'bigReg');
if ~exist(bigparams.REGISTRATION_WORKINGDIR,'dir')
   mkdir(bigparams.REGISTRATION_WORKINGDIR); 
end

bigparams.EXPERIMENT_TILESIZE = yamlspecs.montage_size;
bigparams.NUMTILES = prod(bigparams.EXPERIMENT_TILESIZE);
bigparams.NUMROUNDS = yamlspecs.rounds;
bigparams.DOWNSAMPLE_RATE = yamlspecs.downsample_rate;
%TODO: this is not currently linked to the loadParameters 
bigparams.REFERENCE_ROUND = yamlspecs.ref_round; 

bigparams.FOVS_TO_IGNORE = yamlspecs.fovs_to_skip;


bigparams.TILE_OVERLAP= yamlspecs.fov_overlap;
bigparams.IMG_SIZE_XY = yamlspecs.img_size_xy;

%TODO: integrate these values into the broader yaml usage.
bigparams.REG_CHANNEL = 'summedNorm';
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

