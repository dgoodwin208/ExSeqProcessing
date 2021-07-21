% Author: Zachary Chiang, Buenrostro Lab, Harvard University
% Finds all pixels correlated to a given peak pixel over all images of an 
% IGS experiment and sums signal into a cycles by channels matrix

% Only cosmetic modifications by Dan Goodwin, June 2021
function[mat, spot] = spot_caller(peak, all_stacks, spot_dim, corr_thresh)

    % get sequencing parameters
    num_channels = size(all_stacks,4);
    num_cycles = size(all_stacks,5);
    
    % get area surrounding pixel, adjusting for image bounds
    spot_area = zeros(spot_dim(1)*2+1,spot_dim(2)*2+1,spot_dim(3)*2+1,num_channels,num_cycles);

    x_bounds = [max(1, peak(1)-spot_dim(1)) min(size(all_stacks,1), peak(1)+spot_dim(1))];
    y_bounds = [max(1, peak(2)-spot_dim(2)) min(size(all_stacks,2), peak(2)+spot_dim(2))];
    z_bounds = [max(1, peak(3)-spot_dim(3)) min(size(all_stacks,3), peak(3)+spot_dim(3))];

    x_offsets = [1-(peak(1)-spot_dim(1)-x_bounds(1)) size(spot_area,1)-(peak(1)+spot_dim(1)-x_bounds(2))];
    y_offsets = [1-(peak(2)-spot_dim(2)-y_bounds(1)) size(spot_area,2)-(peak(2)+spot_dim(2)-y_bounds(2))];
    z_offsets = [1-(peak(3)-spot_dim(3)-z_bounds(1)) size(spot_area,3)-(peak(3)+spot_dim(3)-z_bounds(2))];

    spot_area(x_offsets(1):x_offsets(2),y_offsets(1):y_offsets(2),z_offsets(1):z_offsets(2),:,:) = all_stacks(x_bounds(1):x_bounds(2), y_bounds(1):y_bounds(2), z_bounds(1):z_bounds(2),:,:);
    area_pixels = size(spot_area,1)*size(spot_area,2)*size(spot_area,3);
    
    % get correlation of peak pixel to all surrounding pixels over all images
    
    corr_mat = corrcoef(reshape(spot_area,area_pixels,num_channels*num_cycles)');
    spot = reshape(corr_mat((area_pixels-1)/2+1,:),size(spot_area,1),size(spot_area,2),size(spot_area,3));

    % sum signal of all nearby pixels that pass correlation threshold
    
    mat = squeeze(sum(sum(sum((spot_area.*(repmat(spot,1,1,1,4,num_cycles) > corr_thresh)),1),2),3));

end