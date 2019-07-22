%% Define all relevant parameters and known number of tiles per round
% This is the CA1 complete hippocampus acquired on July 9th-July 14th
% Numbers are from metadata and examining the raw data folders

numTiles = 170;
numRows= 10;

imgSizeXY = [2048, 2048];
imgResXY = .1625; % in um, Taken from the .ims properties read via FIJI
imgResZ = 0.3974;

overlap = .02;
outputTileSize = [10,17];
outputRootFolder = '/mp/nas0/ExSeq/Experiments_July2019/CA1-20190714';

xmlfile = '/Users/goody/Neuro/ExSeq/splintr/2019-07-09_19.00.30_Experiment_1LLJGQ2_SplintR_Illumina_072019.xml';
xmlStruct = parseXML(xmlfile);
round_referenceTileStructure = 1;

experiment_name = 'exseqca1rep1';
imgExt = 'h5';

%% Create the spatial map of the tiling snake pattern we want to use

%A map of all the tile indices laid out in a 2D grid
tileMap_indices = [];
for n = 1:numTiles
%     [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
    [row, col] = getTilePositionFromTerastitcherXML(n-1,xmlStruct);
    if row>9 || col > 17
        barf()
    end
    tileMap_indices(row+1,col+1) = n-1;
end

%% Make the complete image, ignoring overlap for now
complete_stitch = zeros(imgSizeXY(1)*size(tileMap_indices,1),...
    imgSizeXY(2)*size(tileMap_indices,2));

for row = 1:size(tileMap_indices_reference,1)
    for col = 1:size(tileMap_indices_reference,2)
        fov_inputnum = tileMap_indices(row,col);
        
        filename_morphology = fullfile(fov_rootfolder,...
                    sprintf('F%.3i',fov_inputnum),...
                    '0_raw',...
                    sprintf('%s-F%.3i_round%.3i_ch%.2i.%s',...
                        experiment_name,fov_outputnum,rnd,c_idx,imgExt) );
        vol = load3DImage_uint16(filename_morphology);
        mip = max(vol,[],3);
        complete_stitch(1+imgSizeXY(1)*(row-1):imgSizeXY(1)*row,...
            1+imgSizeXY(2)*(col-1):imgSizeXY(2)*col) = mip;
    end
end