%% Load the data

load('registration_allkeys_round001.mat')
%%
bigExpParams;
%%
ROUND=1;

numTiles = prod(bigparams.EXPERIMENT_TILESIZE);

%Take an estimate size 
pos = zeros(numTiles*1000,2);

running_ctr = 1;
for F = 1:numTiles
    
    keys = keys_all{F,ROUND};
    
    if length(keys)==0
        continue
    end
    
    for k = 1:length(keys)
        pos(running_ctr,:) = [keys{k}.x_global,keys{k}.y_global];
        running_ctr = running_ctr+1;
    end
    
end
running_ctr

%% Load all the points in space 
% figure;
plot(pos(:,2),pos(:,1),'r.','MarkerSize',15);
% set(gca, 'YDir','reverse')

