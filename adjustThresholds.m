% adjust Threshold instead of using launchThresholdGUI

function adjustThresholds(varargin)

    browsingTools = improc2.launchImageObjectBrowsingTools(varargin{:});

    objectHandle = browsingTools.objectHandle;

    %%
    [rnaChannels, rnaProcessorClassName] = improc2.thresholdGUI.findRNAChannels(objectHandle);

    rnaChannelSwitch = dentist.utils.ChannelSwitchCoordinator(rnaChannels);

    rnaProcessorDataHolder = improc2.utils.ProcessorDataHolder(...
        objectHandle, rnaChannelSwitch, rnaProcessorClassName);
    
    %% models for threshold review and for hasClearThreshold status
    
    if isa(rnaProcessorDataHolder.processorData, 'improc2.interfaces.NodeData')
        % these are new-style image objects

        % make sure there is a thresholdQCData node in every RNA channel
        channelHasQCDataFunc = @(channelName) objectHandle.hasData(...
            channelName, 'improc2.nodeProcs.ThresholdQCData');
        assert(all(cellfun(channelHasQCDataFunc, rnaChannels)), ...
            'All rna channels must have a thresholdQCData data node')
        thresholdQCDataHolder = improc2.utils.ProcessorDataHolder(...
            objectHandle, rnaChannelSwitch, 'improc2.nodeProcs.ThresholdQCData');
        
        thresholdReviewFlagger = improc2.thresholdGUI.ThresholdReviewFlagger(...
            thresholdQCDataHolder);
        clearThresholdProcessorDataHolder = thresholdQCDataHolder;
    else
        % legacy image objects
        thresholdReviewFlagger = improc2.thresholdGUI.NullThresholdReviewFlagger();
        clearThresholdProcessorDataHolder = rnaProcessorDataHolder;
    end
    


    % adjust threshold using Secant method

    % from improc2.utils.NumSpotsTextBox
    maxOfCount = 10;
    x_epsilon = 0.01;
    deriv_epsilon = 1e-14;
    for i=1:browsingTools.navigator.numberOfArrays
        for channelName = rnaChannelSwitch.channelNames
            disp(['[',num2str(i),'] ##### channel=',channelName{1}])
            rnaChannelSwitch.setChannelName(channelName{1});

            % from improc2.thresholdPlugin
            proc = rnaProcessorDataHolder.processorData;
            ranksOfRegionalMaxima = log(numel(proc.regionalMaxValues):-1:1);

            x = proc.regionalMaxValues;
            y = ranksOfRegionalMaxima';
            disp(['size(x)=',num2str(length(x)),',size(y)=',num2str(length(y))])
            dx = (x(end)-x(1))/100;

            ps = dpsimplify([x,y],0.1);
            ns = (ps-min(ps))./(max(ps)-min(ps));
            k = LineCurvature2D(ns);

            k_mid = ceil(length(k)/2);
            [max_k,idx_k] = max(k(1:k_mid));
            x_k = ps(idx_k,1);

            rnaProcessorDataHolder.processorData.threshold = x_k;
            numSpots = rnaProcessorDataHolder.processorData.getNumSpots();
            disp(['[',num2str(i),'] numSpots=',num2str(numSpots), ',threshold=',num2str(x_k)])

            thresholdReviewFlagger.flagThresholdAsReviewed();
        end
        browsingTools.navigator.tryToGoToNextObj()
    end

end

