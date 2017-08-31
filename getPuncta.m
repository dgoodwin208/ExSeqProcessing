%Loop through all the rajlab datastructures and extract the positions of the puncta
loadParameters;
cd(params.punctaSubvolumeDir);
tools = improc2.launchImageObjectTools;

puncta_perchan = cell(params.NUM_ROUNDS,params.NUM_CHANNELS);
while tools.iterator.continueIteration
    disp('Loop')
    if (~tools.annotations.getValue('isGood'))
        tools.iterator.goToNextObject;
        continue;
    end
    
    %Get the filename
    filename = tools.objectHandle.getImageFileName('alexa')
    %Get the three digits and extract the round by Atsushi's format
    %First two digits are the round number, the third digit is the channel
    %number
    exp_idx = str2num(filename(6:7));
    chan_idx = str2num(filename(8))+1;

    objNum = tools.navigator.currentArrayNum();
    [Y X Z] = tools.objectHandle.getData('alexa').getSpotCoordinates();
    
    tools.iterator.goToNextObject()
    puncta_perchan{exp_idx,chan_idx} = [Y X Z];
    exp_idx = exp_idx +1;
end

%For later analysis, save all punta in individual channels
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_allexp_allchans.mat',params.FILE_BASENAME)),'puncta_perchan');

puncta = cell(params.NUM_ROUNDS,1);
for rnd_idx = 1:params.NUM_ROUNDS
    all_locations_per_round = [];
    for c_idx = 1:params.NUM_CHANNELS
        all_locations_per_round = [puncta_perchan{rnd_idx,c_idx}; all_locations_per_round]; 
    end
    puncta{rnd_idx} = all_locations_per_round;
end

save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_allexp.mat',params.FILE_BASENAME)),'puncta');
