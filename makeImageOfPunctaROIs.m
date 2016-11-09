subplot(NUM_ROUNDS,NUM_CHANNELS,1)
filename = 'non-normalized.gif';
%Setting the figure shape so it comes out well in the gif
set(0, 'DefaultFigurePaperPosition', [425   980   576   876]);
figure(1);
for puncta_idx = good_puncta_indices(1:100)
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
%     pause
    
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if puncta_idx == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end