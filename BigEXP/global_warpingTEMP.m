function global_warpingTEMP(round_mov,keys_fixed,pos_fixed,FOVS,FOVS_per_fixed_fov_total,bigparams)

% In this function we know all the moving fovs that are implicated per
% fixed fov. So in this case, we calculate what the warp would be if they
% were directly overlaid. This means we can do the math on local, not
% global parameters. 
mycolors = {'g','m','b','c','k'};color_ctr = 1;
figure('Visible','off')
plot(pos_fixed(:,2),pos_fixed(:,1),'r.','MarkerSize',15);
hold on;

numTiles = prod(size(bigparams.TILE_MAP));
for FOV_fixed = 0:numTiles-1
    fprintf('fixed FOV%.3i matches the following:\n',FOV_fixed);
    
    keys_fixed_subset = keys_fixed(FOVS==FOV_fixed);
    
    
    %Update the color per movingFOV that matches a fixed fov
    color_ctr = mod(color_ctr+1,length(mycolors))+1;
    
    fovs_moving = FOVS_per_fixed_fov_total{FOV_fixed+1,round_mov};
    
    for fov_mov = fovs_moving
        
        fprintf('Matching movFov%.3i to fixedFOV%.3i\n',fov_mov,FOV_fixed);
        
        keys = loadFOVKeyptsAndFeatures(fov_mov,round_mov,bigparams);
        if isempty(keys)
            continue
        end
        keys_moving = cell(length(keys),1);
        for idx = 1:length(keys)
            keys_moving{idx} = keys{idx};
            keys_moving{idx}.x = keys_moving{idx}.x_global;
            keys_moving{idx}.y = keys_moving{idx}.y_global;
            keys_moving{idx}.z = keys_moving{idx}.z_global;
        end
        
        
        try
            [keyM_total,keyF_total] = calcCorrespondences_global(keys_moving,keys_fixed_subset);
            %Calculate the affine tform and get back the transformed keypoints
            [affine_tform,keyM_total_transformed]  = getGlobalAffineFromCorrespondences(keyM_total,keyF_total);
        catch
            fprintf("failed to find sufficient correspondences between FOVS %i and %i, skip!\n",FOV_fixed,fov_mov)
            continue
        end
        
        %calculatinos
        pos_moving = keyM_total_transformed(:,1:2);
        %NOTE the XY swap!!
        plot(pos_moving(:,1),pos_moving(:,2),sprintf('%s.',mycolors{color_ctr}),'MarkerSize',15);
        
        output_keys_filename = fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('fovtransform_fixed%.3i_moving%.3i_round%03d.mat',FOV_fixed,fov_mov,round_mov));
        
        save(output_keys_filename,'keyM_total','keyF_total','affine_tform','keyM_total_transformed');
        
    end %end the loop over the moving fields of view
end %end loop over the rounds

saveas(gcf,fullfile(bigparams.REGISTRATION_WORKINGDIR,sprintf('complete_allRoundMatching_round%.3i.fig',round_mov)))
close

end

