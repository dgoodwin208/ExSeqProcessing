tools = improc2.launchImageObjectTools;
out = [];
while tools.iterator.continueIteration
    if (~tools.annotations.getValue('isGood'))
        tools.iterator.goToNextObject;
        continue;
    end
    
    objNum = tools.navigator.currentArrayNum();
    [Y X Z] = tools.objectHandle.getData('alexa').getSpotCoordinates();
    out = [out; Y X Z ones(length(X),1)*objNum];
    tools.iterator.goToNextObject()
end

save('puncta_coords.mat','X','Y','Z');