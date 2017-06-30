%%
loadParameters;
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));
load(fullfile(params.transcriptResultsDir,sprintf('%s_transcripts_probfiltered.mat',params.FILE_BASENAME)));
load(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptsv9',params.FILE_BASENAME)));

puncta_set = puncta_set(:,:,:,:,:,puncta_indices_probfiltered);
pos = pos(puncta_indices_probfiltered,:);
%%

hasInitGif = 0;
giffilename='puncta_transcripts_exseq.gif';


output_ctr = 1;

indices_to_view = 1:100;

transcripts_to_view = transcripts_probfiltered(indices_to_view,:);
pos_to_view = pos(indices_to_view,:);
punctaset_to_view = puncta_set(:,:,:,:,:,indices_to_view);

fullfile(params.transcriptResultsDir,sprintf('%s_tinyplottablesubset',params.FILE_BASENAME),...
    'transcripts_to_view','pos_to_view','punctaset_to_view');

barf();



for t_idx = 1:size(transcripts_to_view,1)
    
    if(mod(t_idx,100)==0)
        fprintf('Writing Puncta idx %i\n',t_idx)
    end
    
    
    
    figure(1);
    clf('reset')
    ha = tight_subplot(params.NUM_ROUNDS,params.NUM_CHANNELS,zeros(params.NUM_ROUNDS,2)+.01);
    
    subplot_idx = 1;
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,t_idx));
        
        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        values = zeros(4,1);
        
        for c_idx = 1:params.NUM_CHANNELS
            clims = [min_intensity,max_intensity];
            
            %Get the subplot index using the tight_subplot system
            axes(ha(subplot_idx));
            
            punctaVol = squeeze(punctaset_perround(:,:,:,c_idx));
            imagesc(max(punctaVol,[],3),clims);
            
            if c_idx==1 && exp_idx ==1
                axis off;
                position_string = sprintf('(%i,%i,%i)',pos_to_view(t_idx,1),pos_to_view(t_idx,2),pos_to_view(t_idx,3));
                text(-0.0,10.,position_string,'rotation',90)
%                                 ylabel(top_guess_string(exp_idx));
                                axis tight;
                
            else
                axis off;
            end
            
            if exp_idx==1
                title(sprintf('%i',c_idx));
            end
            colormap gray
            subplot_idx = subplot_idx+1;
        end
    end
    
    drawnow
    if hasInitGif==0
        pause
    end
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if hasInitGif==0
        imwrite(imind,cm,giffilename,'gif', 'Loopcount',inf);
        hasInitGif = 1;
    else
        imwrite(imind,cm,giffilename,'gif','WriteMode','append');
    end
    
    
    %         figure(2);
    %
    %         imagesc(maxProj);
    %         hold on;
    %         plot(X(t_idx),Y(t_idx),'r.');
    %         hold off;
    %         axis off;
    %         title(top_guess_string + ' Max intensity for Round5' );
    %         pause
end



disp('Done!')
