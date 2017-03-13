%Run this as a cell from the rajLabTools directory containing the data
loadParameters;
cd(params.rajlabDirectory);
tools = improc2.launchImageObjectTools;
% out = [];
puncta = {}; exp_idx = 1;
while tools.iterator.continueIteration
    disp('Loop')
    if (~tools.annotations.getValue('isGood'))
        tools.iterator.goToNextObject;
        continue;
    end
    
    objNum = tools.navigator.currentArrayNum();
    [Y X Z] = tools.objectHandle.getData('alexa').getSpotCoordinates();
%     out = [out; Y X Z ones(length(X),1)*objNum];
    tools.iterator.goToNextObject()
    puncta{exp_idx} = [Y X Z];
    exp_idx = exp_idx +1;
end

% save('puncta_coords.mat','X','Y','Z');
save('puncta_allexp.mat','puncta');
