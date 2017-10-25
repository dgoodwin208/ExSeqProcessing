function visualizeGridPlot(puncta,transcripts,params,fignum)

    
    figure(fignum);
    
    clf('reset')
    ha = tight_subplot(params.NUM_ROUNDS,params.NUM_CHANNELS+1,zeros(params.NUM_ROUNDS,2)+.01);
    
    subplot_idx = 1;
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta(:,:,:,exp_idx,:));
        
        max_intensity = max(punctaset_perround(:))+1;
        min_intensity = min(punctaset_perround(:));
        
        max_vol_stack = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,4);
        
        for c_idx = 1:params.NUM_CHANNELS
            clims = [min_intensity,max_intensity];
            
            %Get the subplot index using the tight_subplot system
            axes(ha(subplot_idx));
            
            punctaVol = squeeze(punctaset_perround(:,:,:,c_idx));
            
            max_vol_stack(:,:,c_idx) = max(punctaVol,[],3);
%             z_idx = ceil(size(punctaVol,3)/2);
%             imagesc(squeeze(punctaVol(:,:,z_idx)),clims);
            imagesc(max_vol_stack(:,:,c_idx),clims);
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
        
        %Add a fifth column to visualize the puncta in RGB
        %Get the subplot index using the tight_subplot system
        axes(ha(subplot_idx));
        max_vol_stack = uint8(1*(max_vol_stack./max(max_vol_stack(:))));
        rgb_img = makeRGBImageFrom4ChanData(max_vol_stack);
        
        imshow(rgb_img,'InitialMagnification','fit');
        subplot_idx = subplot_idx+1;
    end
    
   

end
