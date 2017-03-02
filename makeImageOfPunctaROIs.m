%The ROIs are loaded from rois_votedandglobalnormalized.mat
% load('rajlab/ExSeqCulture/roi_parameters_and_punctaset.mat')

%The transcripts are loaded from v3transcripts.mat


NUM_ROUNDS = 3;
NUM_CHANNELS = 4;
subplot(NUM_ROUNDS,NUM_CHANNELS,1)
filename = 'splintr_normalized.gif';
%Setting the figure shape so it comes out well in the gif
% set(0, 'DefaultFigurePaperPosition', [425   980   576   876]);

puncta_directory = '/Users/Goody/Neuro/ExSeq/rajlab/splintr1/';
%load sample image for reference
img = load3DTif(fullfile(puncta_directory,'alexa001.tiff'));
% img = load3DTif('/Users/Goody/Neuro/ExSeq/rajlab/splintr1/FULLTPSsplintr1_round3_ch00Norm.tif');

%% To visualize, we have to re-remove the puncta that are on the edge
% (of the space)
load(fullfile(puncta_directory,'rois_votednonnormed.mat'));
load(fullfile(puncta_directory,'transcriptsv4_punctanormed.mat'));
keep_indices = zeros(size(puncta_set,6),1);
keep_indices(good_puncta_indices)=1;
keep_indices = logical(keep_indices);
puncta_set = puncta_set(:,:,:,:,:,keep_indices);

%Similarly, load the puncta_filtered.mat file and filter those positions as
%well

puncta_filtered = puncta_filtered(keep_indices,:);
%unpack the filtered coords into X,Y,Z vectors
Y = round(puncta_filtered(:,1));
X = round(puncta_filtered(:,2));
Z = round(puncta_filtered(:,3));

maxProj = max(img,[],3);

%% 
figure;
imagesc(maxProj);
hold on;
for x = 1:size(puncta_filtered,1)
    plot(X(x),Y(x),'r.');
end
hold off;

%% How does our calculation of confidence vary across the rounds?
confidence_mean = mean(transcripts_confidence,1);
confidence_std = std(transcripts_confidence);

figure
hold on
bar(1:NUM_ROUNDS,confidence_mean)
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

possibles = 1:size(accepted_locations);
possibles =possibles(accepted_locations); 
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