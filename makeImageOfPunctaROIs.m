%The ROIs are loaded from rois_votedandglobalnormalized.mat
% load('rajlab/ExSeqCulture/roi_parameters_and_punctaset.mat')

%The transcripts are loaded from v3transcripts.mat

loadParameters;
subplot(params.NUM_ROUNDS,params.NUM_CHANNELS,1)
filename = 'splintr_normalized.gif';
%Setting the figure shape so it comes out well in the gif
% set(0, 'DefaultFigurePaperPosition', [425   980   576   876]);

% puncta_directory = '/Users/Goody/Neuro/ExSeq/rajlab/splintr1/';
%load sample image for reference
img = load3DTif(fullfile(puncta_directory,'alexa001.tiff'));

maxProj = max(img,[],3);
%% To visualize, we have to re-remove the puncta that are on the edge
% (of the space)
load(fullfile(params.rajlabDirectory,'rois_votednonnormed16b.mat'));
load(fullfile(params.rajlabDirectory,'transcriptsv8_punctameannormed.mat'));


%% 
figure;
imagesc(maxProj);
hold on;
for x = 1:size(X,1)
    if indices_interAndIntraAgreements(x)
        plot(X(x),Y(x),'g.');
    else
        plot(X(x),Y(x),'r.');
    end
    
end
hold off;

%% How does our calculation of confidence vary across the rounds?
confidence_mean = mean(transcripts_confidence,1);
confidence_std = std(transcripts_confidence);

figure
bar(1:NUM_ROUNDS,confidence_mean)
hold on
errorbar(1:NUM_ROUNDS,confidence_mean,confidence_std/sqrt(size(transcripts_confidence,1)),'.')
hold off;
xlim([0,NUM_ROUNDS+1])
xlabel('Sequencing Round');
ylabel(sprintf('Mean confidence measure across %i puncta with SE bars',size(transcripts_confidence,1)))
title('Mean confidence measure across ExSeq rounds');

%Get the median too
figure

bar(1:NUM_ROUNDS,median(transcripts_confidence,1))
xlim([0,NUM_ROUNDS+1])
xlabel('Sequencing Round');
ylabel(sprintf('Median confidence measure across %i puncta',size(transcripts_confidence,1)))
title('Median of confidence measure across ExSeq rounds');

%% Generate a list of puncta for visualization


figure(2);
imagesc(maxProj); 

hasInitGif = 0;

%Shahar's code
% possibles = 1:size(accepted_locations);
% possibles =possibles(accepted_locations); 

%Combing inter and intra color comparisons
possibles = 1:size(indices_interAndIntraAgreements);
possibles =possibles(indices_interAndIntraAgreements); 

to_vizualize = possibles(1:100);

for puncta_idx = to_vizualize
    figure(1);
    fprintf('Original idx: %i\n',possibles(puncta_idx));
    transcript_puncta = transcripts(puncta_idx,:);
    transcriptconfidence_puncta = transcripts_confidence(puncta_idx,:);
    
    
    subplot_idx = 1;
    
    for exp_idx = 1:NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));

        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        values = zeros(4,1);
        for c_idx = 1:NUM_CHANNELS

            clims = [min_intensity,max_intensity];
            subplot(NUM_ROUNDS,NUM_CHANNELS,subplot_idx)
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