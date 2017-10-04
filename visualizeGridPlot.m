function visualizeGridPlot(puncta,transcripts,params,fignum)

    
    figure(fignum);
    
    clf('reset')
    ha = tight_subplot(params.NUM_ROUNDS,params.NUM_CHANNELS,zeros(params.NUM_ROUNDS,2)+.01);
    
    subplot_idx = 1;
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta(:,:,:,exp_idx,:));
        
        max_intensity = max(punctaset_perround(:))+1;
        min_intensity = min(punctaset_perround(:));
        
        for c_idx = 1:params.NUM_CHANNELS
            clims = [min_intensity,max_intensity];
            
            %Get the subplot index using the tight_subplot system
            axes(ha(subplot_idx));
            
            punctaVol = squeeze(punctaset_perround(:,:,:,c_idx));
            
%             z_idx = ceil(size(punctaVol,3)/2);
%             imagesc(squeeze(punctaVol(:,:,z_idx)),clims);
            imagesc(max(punctaVol,[],3),clims);
            axis off;
            if numel(transcripts)>1 && c_idx==transcripts(exp_idx)
                title(sprintf('%i',c_idx),'Color','m')
            end
            if c_idx==1 && exp_idx ==1
                
                position_string = '';
                text(-0.0,10.,position_string,'rotation',90)
                 axis tight;
                
            else
                axis off;
            end
            
            colormap gray
            subplot_idx = subplot_idx+1;
        end
    end
    
   

end
