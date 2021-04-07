% Load the file created by completionReport and visualize the results
function completionReport_visualize(yamlfile, compReport_file)

%Load the expResults struct
load(compReport_file);

%Use a YAML file to load the parameters for a large experiment
yamlspecs = ReadYaml(yamlfile);
%The YAML files might have extra quotes so remove them
BASENAME = strrep(yamlspecs.basename,'''','');
outputTileSize = yamlspecs.montage_size;

numRows= outputTileSize(1);
numTiles = prod(outputTileSize);
tileMap_indices_reference = [];

for n = 1:numTiles
    [row, col] = calculateRowColForSnakeTilePattern(n-1,numRows);    
    tileMap_indices_reference(row+1,col+1) = n-1;
end

map_puncta_intact = zeros(size(tileMap_indices_reference));
map_percVolUsed = zeros(size(tileMap_indices_reference));
for row = 1:size(tileMap_indices_reference,1)
    for col = 1:size(tileMap_indices_reference,2)
        
        fov_inputnum = tileMap_indices_reference(row,col);
        expR = expResults{fov_inputnum+1};
        
        map_puncta_intact(row,col) = expR.num_complete_puncta;
        map_percVolUsed(row,col) = expR.percentage_volume_usable;
        
    end
end

figure;
subplot(1,2,1);
imagesc(map_puncta_intact);
for row = 1:size(tileMap_indices_reference,1)
    for col = 1:size(tileMap_indices_reference,2)
        if map_percVolUsed(row,col)>0
            text(col-.25,row,sprintf('%i',map_puncta_intact(row,col)))
        end
    end
end
title(sprintf('%i usable puncta in %s',sum(map_puncta_intact(:)),BASENAME))
subplot(1,2,2);
imagesc(map_percVolUsed);
for row = 1:size(tileMap_indices_reference,1)
    for col = 1:size(tileMap_indices_reference,2)
        if map_percVolUsed(row,col)>0
            text(col-.25,row,sprintf('%.2f',map_percVolUsed(row,col)))
        end
    end
end
title(sprintf('Percentage usable image volume %s',BASENAME))

end
