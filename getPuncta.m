%Loop through all the rajlab datastructures and extract the positions of the puncta
loadParameters;
cd(params.punctaSubvolumeDir);
tools = improc2.launchImageObjectTools;

puncta = {}; exp_idx = 1;
while tools.iterator.continueIteration
    disp('Loop')
    if (~tools.annotations.getValue('isGood'))
        tools.iterator.goToNextObject;
        continue;
    end
    
    objNum = tools.navigator.currentArrayNum();
    [Y X Z] = tools.objectHandle.getData('alexa').getSpotCoordinates();
    
    tools.iterator.goToNextObject()
    puncta{exp_idx} = [Y X Z];
    exp_idx = exp_idx +1;
end

save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_allexp.mat',params.FILE_BASENAME)),'puncta');
