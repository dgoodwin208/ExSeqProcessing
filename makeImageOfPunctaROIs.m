%The ROIs are loaded from rois_votedandglobalnormalized.mat
% load('rajlab/ExSeqCulture/roi_parameters_and_punctaset.mat')

%The transcripts are loaded from transcripts_center-only.mat

good_puncta_indices = setdiff(1:num_puncta,bad_puncta_indices);

subplot(NUM_ROUNDS,NUM_CHANNELS,1)
filename = 'normalized.gif';
%Setting the figure shape so it comes out well in the gif
set(0, 'DefaultFigurePaperPosition', [425   980   576   876]);

%% How does our calculation of confidence vary across the rounds?
confidence_mean = mean(transcripts_confidence,1);
confidence_std = std(transcripts_confidence);

figure
hold on
bar(1:12,confidence_mean)
errorbar(1:12,confidence_mean,confidence_std/sqrt(size(transcripts_confidence,1)),'.')
hold off;
xlim([0,13])
xlabel('Sequencing Round');
ylabel(sprintf('Mean confidence measure across %i puncta with SE bars',size(transcripts_confidence,1)))
title('Statistics of confidence measure across ExSeq rounds');

%Get the median too
figure

bar(1:12,median(transcripts_confidence,1))
xlim([0,13])
xlabel('Sequencing Round');
ylabel(sprintf('Median confidence measure across %i puncta',size(transcripts_confidence,1)))
title('Median of confidence measure across ExSeq rounds');

%% Generate a list of puncta for visualization

puncta_to_viz = [];

%%
hasInitGif = 0;
for puncta_idx = good_puncta_indices(1:100)
    figure(1);
    
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
    
    figure(2)
    pause
    
end