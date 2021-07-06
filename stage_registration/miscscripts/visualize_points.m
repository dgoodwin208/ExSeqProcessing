% In the case of debugging a failed registration, it can be helpful to
% visually inspect the spatial distribution of the matching keypoints
% between two rounds


loadParameters;

%Choose a round that you want to visualize, make sure it's not the
%reference round!
ROUND_TO_EXAMINE = 3;
if params.REFERENCE_ROUND_WARP == ROUND_TO_EXAMINE
    error('Choose a round other than the reference round to visualize');
end
if params.DO_DOWNSAMPLE
        filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
    else
        filename_root = sprintf('%s_',params.FILE_BASENAME);
end
    
% LOAD KEYS
output_keys_filename = fullfile(params.registeredImagesDir,...
    sprintf('globalkeys_%sround%.3d.mat',filename_root,ROUND_TO_EXAMINE));
load(output_keys_filename);

%Create  plot of all the feature + correspondences
figure;

plot3(keyF_total(:,1),keyF_total(:,2),keyF_total(:,3),'o');
hold on;
for k_idx=1:size(keyF_total,1)
    plot3(keyM_total(k_idx,1),keyM_total(k_idx,2),keyM_total(k_idx,3),'ro');
    lines = [ ...
            [keyM_total(k_idx,1);keyF_total(k_idx,1)] ... 
            [keyM_total(k_idx,2);keyF_total(k_idx,2)] ...
            [keyM_total(k_idx,3);keyF_total(k_idx,3)] ];
     
     rgb = [0 0 0];
     if lines(1,1) > lines(2,1)
         rgb(1) = .7;
     end
     if lines(1,2) > lines(2,2)
         rgb(2) = .7;
     end
     if lines(1,3) > lines(2,3)
         rgb(3) = .7;
     end
     plot3(lines(:,1),lines(:,2),lines(:,3),'color',rgb);   
end
legend('Fixed', 'Moving');
title(sprintf('%i correspondences to calculate Registration warp',size(keyF_total,1)))
view(45,45);
