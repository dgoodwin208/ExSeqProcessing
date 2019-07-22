%Used to copy all data from the complete CA-1 morphology round into h5
%files in an organized folder structure. Specifically, this script must
%also create a common field of view numbering structure, which is important
%because a microscope glitch created a varying number of tiles per
%sequencing round
% Written by Dan Goodwin, dgoodwin@mit.edu July 20 2019

%% Define all relevant parameters and known number of tiles per round
% This is the CA1 complete hippocampus acquired on July 9th-July 14th
% Numbers are from metadata and examining the raw data folders
imsfolders_perRound = ...
    {'/mp/nas0/B46Cambridge/data/data_SplintR0719/2019-07-09',...
     '/mp/nas0/B46Cambridge/data/data_SplintR0719/2019-07-10',...
     '/mp/nas0/B46Cambridge/data/data_SplintR0719/2019-07-11',...
     '/mp/nas0/B46Cambridge/data/data_SplintR0719/2019-07-12',...
     '/mp/nas0/B46Cambridge/data/data_SplintR0719/2019-07-14'};
    
numTiles_perRound = [170,170,198,180,180];
numRows_perRound = [10,10,11,10,10];

imgSizeXY = [2048, 2048];
imgResXY = .1625; % in um, Taken from the .ims properties read via FIJI
imgResZ = 0.3974;

outputTileSize = [10,17];
outputRootFolder = '/mp/nas0/ExSeq/Experiments_July2019/CA1-20190714';

round_referenceTileStructure = 1;

imgExt = 'tif'; %tif or h5

experiment_name = 'exseqca1rep1';
%% Create the spatial map of the tiling snake pattern we want to use 

%A map of all the tile indices laid out in a 2D grid 
tileMap_indices_reference = [];

numTiles = numTiles_perRound(round_referenceTileStructure);
numRows = numRows_perRound(round_referenceTileStructure);
for n = 1:numTiles
    [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
    tileMap_indices_reference(row+1,col+1) = n-1;
end

%% Loop over all rounds and copy the files into the proper folder structure
bad_tiles = cell(length(imsfolders_perRound),1);
parpool(5);
parfor rnd = 1:length(imsfolders_perRound)
    
    %A map of all the tile indices laid out in a 2D grid 
    tileMap_indices = [];

    numTiles = numTiles_perRound(rnd);
    numRows = numRows_perRound(rnd);
    
    % Create the 2D map of this round's tiling
    for n = 1:numTiles
        [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
        tileMap_indices(row+1,col+1) = n-1;
    end

    %Initialize the counter for bad tiles 
    bad_tiles{rnd} = [];
    %Loop over the reference tile structure, extract the IMS file using the
    %ImarisReader class, taken from https://github.com/PeterBeemiller/ImarisReader
    %and create the ExSeqProcessing folder structure for that FoV
    %NOTE: This assumes that the top left corner of the different tiled
    %images is the same
    for row = 1:size(tileMap_indices_reference,1)
        for col = 1:size(tileMap_indices_reference,2)
            fov_outputnum = tileMap_indices_reference(row,col);
            fov_inputnum = tileMap_indices(row,col);
        
            %Are the folders set up for the ExSeqProcessing?
            fov_rootfolder = fullfile(outputRootFolder,sprintf('F%.3i',fov_outputnum));
            if ~exist(fov_rootfolder,'dir')
                mkdir(fov_rootfolder);
                mkdir(fullfile(fov_rootfolder,'0_raw'));
                mkdir(fullfile(fov_rootfolder,'1_deconvolution'));
                mkdir(fullfile(fov_rootfolder,'2_color-correction'));
                mkdir(fullfile(fov_rootfolder,'3_normalization'));
                mkdir(fullfile(fov_rootfolder,'4_registration'));
                mkdir(fullfile(fov_rootfolder,'5_puncta-extraction'));
                mkdir(fullfile(fov_rootfolder,'6_base-calling'));
            end
            
            %Load the ims file
            inputfolder = imsfolders_perRound{rnd};
            %The actual file names vary greatly, but they all finish with
            %the tilenumber, ie, F030.ims for the 31st tile (0-indexed)
            fileInfo = dir(fullfile(inputfolder,sprintf('*_F%.3i.ims',fov_inputnum)));
            
            %Load the Imaris Object using the Imaris Reader
            try
            	imarisObj = ImarisReader(fullfile(inputfolder,fileInfo.name));
            catch
		fprintf('ERROR Something wrong with file %s\n',fullfile(inputfolder,fileInfo.name));
		bad_tiles{rnd}(end+1) = fov_inputnum;
		continue
	    end
            % Loop over all the channels in the datafile
            numChannels = length(imarisObj.DataSet.ChannelInfo);
            
            fprintf('\tConverting %i channels for file %s\n',numChannels,...
                fullfile(inputfolder,fileInfo.name));
            for c_idx = 0:numChannels-1 %ImarisReader is 0-indexed 
                %GetDataVolume loads channel and time index, time index is
                %always zero for ExSeq data, so we just loop over the
                %channel
                outputfilename = fullfile(fov_rootfolder,'0_raw',...
                    sprintf('%s-F%.3i_round%.3i_ch%.2i.%s',...
                    experiment_name,fov_outputnum,rnd,c_idx,imgExt) );
                if ~exist(outputfilename,'file')
                    vol = imarisObj.DataSet.GetDataVolume(c_idx,0);
		    save3DImage_uint16(vol,outputfilename);
		else
		    fprintf('\tSkipping file %s that already exists\n',outputfilename); 
		end
            end
        end
    end
end
