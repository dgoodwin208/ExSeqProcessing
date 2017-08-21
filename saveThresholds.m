% save Threshold instead of using launchThresholdGUI

function saveThresholds(varargin)

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
    


    % from improc2.utils.NumSpotsTextBox

    %Make a permutation for the loop on the i vector:
    %Whichever round will be the refence for the rajlab must be done first
    loadParameters;
    narrays = 1:browsingTools.navigator.numberOfArrays;
    narrays([1,params.REFERENCE_ROUND_PUNCTA])=[narrays(params.REFERENCE_ROUND_PUNCTA) narrays(1)];

    for i=narrays
        for channelName = rnaChannelSwitch.channelNames
            disp(['[',num2str(i),'] ##### channel=',channelName{1}])
            rnaChannelSwitch.setChannelName(channelName{1});

            % from improc2.thresholdPlugin
            proc = rnaProcessorDataHolder.processorData;
            ranksOfRegionalMaxima = log(numel(proc.regionalMaxValues):-1:1);

            numSpots = rnaProcessorDataHolder.processorData.getNumSpots();
            threshold = rnaProcessorDataHolder.processorData.threshold;
            fprintf('[%i] numSpot=%6i,threshold=%14.3f\n',i,numSpots,threshold)

            thresholdReviewFlagger.flagThresholdAsReviewed();
        end
        browsingTools.navigator.tryToGoToNextObj()
    end

end

