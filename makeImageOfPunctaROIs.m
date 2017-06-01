%The ROIs are loaded from rois_votedandglobalnormalized.mat

%The transcripts are loaded from v3transcripts.mat

loadParameters;

filename = 'splintr_100samples.gif';
%Setting the figure shape so it comes out well in the gif
% set(0, 'DefaultFigurePaperPosition', [425   980   576   876]);

%load sample image for reference
img = load3DTif(fullfile(params.punctaSubvolumeDir,'alexa001.tiff'));

maxProj = max(img,[],3);
%% To visualize, we have to re-remove the puncta that are on the edge
% (of the space)
load(fullfile(params.punctaSubvolumeDir,'rois_votednonnormed16b.mat'));
load(fullfile(params.punctaSubvolumeDir,'transcriptsv8_punctameannormed.mat'));


%% 
figure;
imagesc(maxProj);
hold on;
%Loop over every puncta that passed the spatial filtering step
for x = 1:size(X,1)
    %If successfully agreed on both inter and intra 
    if indices_interAndIntraAgreements(x)
        plot(X(x),Y(x),'g.');
    else
    %If otherwise filtered
        plot(X(x),Y(x),'r.');
    end
    
end
axis off;
hold off;

%% How does our calculation of confidence vary across the rounds?
confidence_mean = mean(transcripts_probfiltered_confidence,1);
confidence_std = std(transcripts_probfiltered_confidence);

figure
subplot(1,2,1);
bar(1:params.NUM_ROUNDS,confidence_mean)
hold on
errorbar(1:params.NUM_ROUNDS,confidence_mean,confidence_std/sqrt(size(transcripts_probfiltered_confidence,1)),'.')
hold off;
xlim([0,params.NUM_ROUNDS+1])
xlabel('Sequencing Round');
ylabel(sprintf('Mean confidence measure across %i puncta with SE bars',size(transcripts_probfiltered_confidence,1)))
title('Mean confidence measure across ExSeq rounds');

%Get the median too
% figure
subplot(1,2,2);
bar(1:params.NUM_ROUNDS,median(transcripts_probfiltered_confidence,1))
xlim([0,params.NUM_ROUNDS+1])
xlabel('Sequencing Round');
ylabel(sprintf('Median confidence measure across %i puncta',size(transcripts_probfiltered_confidence,1)))
title('Median of confidence measure across ExSeq rounds');

%% Generate a list of puncta for visualization


figure(2);
imagesc(maxProj); 

hasInitGif = 0;

%Shahar's code
% possibles = 1:size(accepted_locations);
% possibles =possibles(accepted_locations); 

%The indices to the puncta have been noted in puncta_indices_probfiltered
possibles = puncta_indices_probfiltered;
% possibles =possibles(indices_interAndIntraAgreements); 

% to_vizualize = possibles(1:100);
% transcript_indices = 1:100
to_visualize = 1:100;
for transcript_idx = to_visualize
    puncta_idx = puncta_indices_probfiltered(transcript_idx);
    
    figure(1);
    fprintf('Original idx: %i\n',possibles(transcript_idx));
    transcript_puncta = transcripts_probfiltered(transcript_idx,:);
    transcriptconfidence_puncta = transcripts_probfiltered_confidence(transcript_idx,:);
    
    
    subplot_idx = 1;
    
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));

        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        values = zeros(4,1);
        for c_idx = 1:params.NUM_CHANNELS

            clims = [min_intensity,max_intensity];
            subplot(params.NUM_ROUNDS,params.NUM_CHANNELS,subplot_idx)
            data = squeeze(punctaset_perround(:,:,:,c_idx));
            imagesc(max(data,[],3),clims);

            if(c_idx==transcript_puncta(exp_idx))
                title(sprintf('Conf:%d', transcriptconfidence_puncta(exp_idx)));
            end
            
            axis off; colormap gray
            subplot_idx = subplot_idx+1;
        end
    end
    
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if hasInitGif==0
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        hasInitGif = 1;
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
%     figure(2);
%     
%     imagesc(maxProj);
%     hold on;
%     plot(X(puncta_idx),Y(puncta_idx),'r.');
%     hold off;
%     
%     
%     pause
    
end
