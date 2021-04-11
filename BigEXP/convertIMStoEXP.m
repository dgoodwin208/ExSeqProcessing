function convertIMStoEXP(yamlfile)
%CONVERTIMSTOEXP Convert the IMS files from a multi-view acquisition using
%the YAML file for an experiment.

%Use a YAML file to load the parameters for a large experiment
yamlspecs = ReadYaml(yamlfile);

%The YAML files might have extra quotes so remove them
EXPERIMENT_NAME = strrep(yamlspecs.basename,'''','');
OUTPUTTILESIZE = yamlspecs.montage_size;

NUM_ROUNDS = yamlspecs.rounds;
OUTPUTROOTFOLER = strrep(yamlspecs.base_dir,'''','');

%It takes a bit of hacking to load a list of strings from a yaml 
%This is loading the list of folders 
imsfolders_perRound_temp = strrep(yamlspecs.imsfolders_perround,'''','');
imsfolders_perRound_temp = strrep(imsfolders_perRound_temp,'[','');
imsfolders_perRound_temp = strrep(imsfolders_perRound_temp,']','');
imsfolders_perRound_temp = split(imsfolders_perRound_temp,',');
IMSSFOLDERSPERROUND = cell(length(imsfolders_perRound_temp),1);
for c = 1:length(imsfolders_perRound_temp)
    IMSSFOLDERSPERROUND{c} = imsfolders_perRound_temp{c};
end
        
numTiles_perRound = ones(NUM_ROUNDS,1)*prod(OUTPUTTILESIZE);
numRows_perRound = ones(NUM_ROUNDS,1)*OUTPUTTILESIZE(1);

ROUND_REFERENCETILESTRUCTURE = 1;

IMGEXT = 'h5'; %tif or h5

%When we load the FOV files, we need to know if we have more or less than
%100 .ims files, because the Andor microscope outputs either F00 or F000.
isTwoDigitFOVinputnum = false;
numFOVS = length(dir(fullfile(IMSSFOLDERSPERROUND{ROUND_REFERENCETILESTRUCTURE},'*.ims')));
if numFOVS<100
    isTwoDigitFOVinputnum = true;
end
    
    
%% Create the spatial map of the tiling snake pattern we want to use

%A map of all the tile indices laid out in a 2D grid
tileMap_indices_reference = [];

numTiles = numTiles_perRound(ROUND_REFERENCETILESTRUCTURE);
numRows = numRows_perRound(ROUND_REFERENCETILESTRUCTURE);
for n = 1:numTiles
    [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
    tileMap_indices_reference(row+1,col+1) = n-1;
end

%% Loop over all rounds and copy the files into the proper folder structure
bad_tiles = cell(length(IMSSFOLDERSPERROUND),1);
parpool(4);
parfor rnd = 1:length(IMSSFOLDERSPERROUND)
    
    %A map of all the tile indices laid out in a 2D grid
    tileMap_indices = [];
    
    numTiles = numTiles_perRound(rnd);
    numRows = numRows_perRound(rnd);
    
    % Create the 2D map of this round's tiling
    for n = 1:numTiles
        [rowx, colx] = calculateRowColForSnakeTilePattern(n-1,numRows);
        tileMap_indices(rowx+1,colx+1) = n-1;
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
            fov_rootfolder = fullfile(OUTPUTROOTFOLER,sprintf('F%.3i',fov_outputnum));
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
            inputfolder = IMSSFOLDERSPERROUND{rnd};
            %The actual file names vary greatly, but they all finish with
            %the tilenumber, ie, F030.ims for the 31st tile (0-indexed)
            if isTwoDigitFOVinputnum 
                fileInfo = dir(fullfile(inputfolder,sprintf('*_F%.2i.ims',fov_inputnum)));
            else
                fileInfo = dir(fullfile(inputfolder,sprintf('*_F%.3i.ims',fov_inputnum)));
            end
            
            %Load the Imaris Object using the Imaris Reader
            try
                fprintf('Loading file %s\n',fullfile(inputfolder,fileInfo.name));
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
                    EXPERIMENT_NAME,fov_outputnum,rnd,c_idx,IMGEXT) );
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
bad_tiles

end

