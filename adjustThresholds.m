% adjust Threshold instead of using launchThresholdGUI

function adjustThresholds(skip_first,varargin)

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
    maxOfCount = 5000;
    targetNumSpots = 0;
    targetThreshold = 0;
    d_threshold = 0.05;

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

            if (i == params.REFERENCE_ROUND_PUNCTA)
                if skip_first == true
                    disp('use predefined numSpots')

                    x_k = rnaProcessorDataHolder.processorData.threshold;
                else
                    % adjust threshold using maximum of curvature in x-y curves
                    disp('adjustment by max of curvature')
                    x = proc.regionalMaxValues;
                    y = ranksOfRegionalMaxima';
                    disp(['size(x)=',num2str(length(x)),',size(y)=',num2str(length(y))])
                    dx = (x(end)-x(1))/100;

                    ps = dpsimplify([x,y],0.1);
                    ns = (ps-min(ps))./(max(ps)-min(ps));
                    k = LineCurvature2D(ns);

                    x_k = 0;
                    k_first = 1;
                    k_mid = ceil(length(k)/2);
                    while x_k == 0
                        [max_k,idx_k] = max(k(k_first:k_mid));
                        x_k = ps(idx_k,1);
                        k_first = idx_k+1;
                    end

                    rnaProcessorDataHolder.processorData.threshold = x_k;
                end

                if params.THRESHOLD_MARGIN > 0
                    x_k = x_k+params.THRESHOLD_MARGIN;
                    rnaProcessorDataHolder.processorData.threshold = x_k;
                end

                numSpots = rnaProcessorDataHolder.processorData.getNumSpots();
                disp(['[',num2str(i),'] numSpots=',num2str(numSpots), ',threshold=',num2str(x_k)])

                targetNumSpots = numSpots;
                targetThreshold = x_k;
            else
                % adjust threshold using Secant method
                disp('adjustment by # of spots')

                numSpots = rnaProcessorDataHolder.processorData.getNumSpots();
                threshold = rnaProcessorDataHolder.processorData.threshold;
                disp(sprintf('[%i] (%3i) numSpot=%6i,threshold=%14.3f',i,0,numSpots,threshold))

                if (numSpots == targetNumSpots)
                    disp(['[',num2str(i),'] found target!'])
                    continue
                end

                count = 1;
                threshold0 = threshold;
                numSpots0 = numSpots;
                threshold1 = targetThreshold;
                while (count <= maxOfCount)
                    rnaProcessorDataHolder.processorData.threshold = threshold1;
                    numSpots1 = rnaProcessorDataHolder.processorData.getNumSpots();
                    disp(sprintf('[%i] (%3i) numSpot=%6i,threshold=%14.3f',i,count,numSpots1,threshold1))

                    if (numSpots1 == targetNumSpots)
                        disp(['[',num2str(i),'] found target!'])
                        break
                    end
                    if (numSpots0 == numSpots1)
                        if (targetNumSpots ~= numSpots1)
                            %disp(['[',num2str(i),'] f'' becomes zero, but numSpots is different from the target.'])
			    if (targetNumSpots < numSpots1)
			        threshold = threshold1*(1.0+d_threshold);
			    else
			        threshold = threshold1*(1.0-d_threshold);
			    end

                            threshold0 = threshold1;
                            threshold1 = threshold;
                            numSpots0 = numSpots1;

                            count = count+1;
                            continue;
                        else
                            disp(['[',num2str(i),'] f'' becomes zero, so iteration is finished.'])
                            break
                        end
                    end

                    threshold = threshold1-(numSpots1-targetNumSpots)*(threshold1-threshold0)/(numSpots1-numSpots0);

                    threshold0 = threshold1;
                    threshold1 = threshold;
                    numSpots0 = numSpots1;

                    count = count+1;
                end

                disp(['[',num2str(i),'] numSpots=',num2str(numSpots1), ',threshold=',num2str(threshold1)])
            end

            thresholdReviewFlagger.flagThresholdAsReviewed();
        end
        browsingTools.navigator.tryToGoToNextObj()
    end

end

