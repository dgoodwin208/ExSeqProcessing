% loadParameters;
%set a yamlfile variable then run this script.
bigparams = bigExpParams(yamlfile);


%% Define all relevant parameters and known number of tiles per round
% This is the CA1 complete hippocampus acquired on July 9th-July 14th
% Numbers are from metadata and examining the raw data folders


imgSizeXY = bigparams.IMG_SIZE_XY;

overlap = bigparams.TILE_OVERLAP;
outputTileSize = bigparams.EXPERIMENT_TILESIZE;
numRows= outputTileSize(1);
numTiles = prod(outputTileSize);
OUTPUTROOTFOLDER = bigparams.EXPERIMENT_FOLDERROOT;

EXPERIMENT_NAME= bigparams.EXPERIMENT_NAME;
IMGEXT = bigparams.IMAGE_EXT;


DOWNSAMPLE_RATE = bigparams.DOWNSAMPLE_RATE;

imgResXY = .1625; % in um, Taken from the .ims properties read via FIJI
imgResZ = 0.3974;



%% Create the spatial map of the tiling snake pattern we want to use

%A map of all the tile indices laid out in a 2D grid
tileMap_indices_reference = [];
for n = 1:numTiles
    [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
    
    tileMap_indices_reference(row+1,col+1) = n-1;
end
%% NEW - instead of flip the images, flip the code!
tileMap_indices_reference = fliplr(transpose(tileMap_indices_reference));

%%
for ROUND = 1:bigparams.NUMROUNDS
    
    %% Make the complete image, ignoring overlap for now
    complete_stitch = zeros(imgSizeXY(1)*size(tileMap_indices_reference,1),...
        imgSizeXY(2)*size(tileMap_indices_reference,2));
    
    for row = 1:size(tileMap_indices_reference,1)
        for col = 1:size(tileMap_indices_reference,2)
            tic
            fov_inputnum = tileMap_indices_reference(row,col);
            
            fov_rootfolder = fullfile(OUTPUTROOTFOLDER,'processing');
            try
                filename_morphology = fullfile(fov_rootfolder,...
                    sprintf('F%.3i',fov_inputnum),...
                    '4_registration',...
                    sprintf('%s-F%.3i_round%.3i_%s_affine.%s',...
                    EXPERIMENT_NAME,fov_inputnum,ROUND,'summedNorm',IMGEXT) );
                vol = load3DImage_uint16(filename_morphology);
                mip = max(vol,[],3);
            catch
                fprintf('Skipping missing file: %s\n',filename_morphology);
                mip = zeros(imgSizeXY);
            end
            
            % Don't do any modifications!
            %mip = flipud(transpose(squeeze(mip)));
            
            complete_stitch(1+imgSizeXY(1)*(row-1):imgSizeXY(1)*row,...
                1+imgSizeXY(2)*(col-1):imgSizeXY(2)*col) = mip;
            
            fprintf('Loaded file F%.3i, placing mip in\trow=%.2i\tcol=%.2i\n',...
                fov_inputnum,row,col);
            toc
            clear vol mip
        end
    end
    
    %% Save the resul
    
    outputfile_ds = fullfile(OUTPUTROOTFOLDER,sprintf('%s_NATURALds%i_round%.3i_%s_affine.tif',...
        EXPERIMENT_NAME,DOWNSAMPLE_RATE,ROUND,'summedNorm'));
    
    save3DImage_uint16(imresize3(complete_stitch,1/DOWNSAMPLE_RATE,'linear'),outputfile_ds);
end