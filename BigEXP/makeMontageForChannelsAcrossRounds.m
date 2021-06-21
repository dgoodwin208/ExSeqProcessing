function makeMontageForChannelsAcrossRounds(channels,yamlfile,saveNonDownsampled)

if nargin<3
    saveNonDownsampled = false;
end

%Use a YAML file to load the parameters for a large experiment
yamlspecs = ReadYaml(yamlfile);
%The YAML files might have extra quotes so remove them
EXPERIMENT_NAME = strrep(yamlspecs.basename,'''','');
outputTileSize = yamlspecs.montage_size;

NUM_ROUNDS = yamlspecs.rounds;
OUTPUTROOTFOLDER = strrep(yamlspecs.base_dir,'''','');

% TODO: Not including the overlap was a mistake we made for the HTAPP
% samples but this needs to be included in future samples
overlap = yamlspecs.fov_overlap; 
% Define all other image parameters
imgSizeXY = [2048, 2048];
numTiles = prod(outputTileSize);
numRows= outputTileSize(1);
IMGEXT = 'h5';

DOWNSAMPLE_RATE = 5.;

for ROUND = 1:NUM_ROUNDS
    for COLCHAN = channels
        
        
        % Create the spatial map of the tiling snake pattern we want to use
        
        %A map of all the tile indices laid out in a 2D grid
        tileMap_indices_reference = [];
        for n = 1:numTiles
            [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);
            
            tileMap_indices_reference(row+1,col+1) = n-1;
        end
        
        % Make the complete image, ignoring overlap for now
        complete_stitch = zeros(...
            floor(imgSizeXY(1)*size(tileMap_indices_reference,1)*(1-overlap)),...
            floor(imgSizeXY(2)*size(tileMap_indices_reference,2)*(1-overlap)));
        
        for row = 1:size(tileMap_indices_reference,1)
            for col = 1:size(tileMap_indices_reference,2)
                tic
                fov_inputnum = tileMap_indices_reference(row,col);
                
                fov_rootfolder = OUTPUTROOTFOLDER;
                try
                    filename_morphology = fullfile(fov_rootfolder,...
                        sprintf('F%.3i',fov_inputnum),...
                        '0_raw',...
                        sprintf('%s-F%.3i_round%.3i_ch%.2i.%s',...
                        EXPERIMENT_NAME,fov_inputnum,ROUND,COLCHAN,IMGEXT) );
                    vol = load3DImage_uint16(filename_morphology);
                    mip = max(vol,[],3);
                catch
                    fprintf('Skipping missing file: %s\n',filename_morphology);
                    mip = zeros(imgSizeXY);
                end
                
                %We know that there are two operations necessary to get into the coordinate space
                %from the Andor software: there is an XY<>YX axis swap (transpose) and then a y-axis
                %flip. It is corrected with this flipud(mip') operations
                mip = flipud(transpose(squeeze(mip)));
                
                start_y = 1+floor(imgSizeXY(1)*(row-1)*(1-overlap))
                start_x = 1+floor(imgSizeXY(2)*(col-1)*(1-overlap))
                
                complete_stitch(start_y:start_y+imgSizeXY(1)-1,...
                                start_x:start_x+imgSizeXY(2)-1 ) = mip;
                
                fprintf('Loaded file F%.3i, placing mip in\trow=%.2i\tcol=%.2i\n',...
                    fov_inputnum,row,col);
                toc
                clear vol mip
            end
        end
        
        % Save the downsampled image
        outputfile_ds = fullfile(OUTPUTROOTFOLDER,sprintf('%s_ds%i_round%.3i_ch%.2i.tif',...
            EXPERIMENT_NAME,DOWNSAMPLE_RATE,ROUND,COLCHAN));
        save3DImage_uint16(imresize3(complete_stitch,1/DOWNSAMPLE_RATE,'linear'),outputfile_ds);
        
        % Save the non-downsampled result if requested
        if saveNonDownsampled
            outputfile = fullfile(OUTPUTROOTFOLDER,sprintf('%s_fullres_round%.3i_ch%.2i.tif',...
                EXPERIMENT_NAME,ROUND,COLCHAN));
            save3DImage_uint16(complete_stitch,outputfile);
        end
    end
    
end
