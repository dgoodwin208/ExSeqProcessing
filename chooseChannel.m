function [ chanmax,confidence ] = chooseChannel(punctaset_perround, num_channels, max_distance)
%CHOOSECHANNEL Basic method: get max value of peak closest to center
%   Detailed explanation goes here

if nargin <3
    max_distance = 100;
end

DISTANCE_BLOWUP = 1000;
region_size = size(punctaset_perround,1);
center_point = [(region_size+1)/2, (region_size+1)/2];

vote_value = zeros(4,1);


for c_idx = 1:num_channels
    
    data = squeeze(punctaset_perround(:,:,:,c_idx));
    
    img = max(data,[],3);
    
    if sum(img(:))==0
        vote_value(c_idx) = 0;
        disp('Empty puncta region');
        continue
    end
    
    [xymax,imax,xymin,imin] = extrema2(img);
    %If no peaks, just give it a zero for that chan peak value
    if length(imax)<1
        vote_value(c_idx) = 0;
        continue
    end
    [x,y] = ind2sub(size(img),imax);
    
    %Calculate the distances of all the extreme from the midpoint
    c = [(x - center_point(1)),(y-center_point(2))];
    distances = diag(sqrt(c*c'));
    
    %enforce the given max_distance from the centroid by "blowing up" any
    %distances that are not
    distances(distances>max_distance) = DISTANCE_BLOWUP;
    
    %Sort them in ascending order to get the clostest point to the middle
    [~,I] = sort(distances,'ascend');
    if distances(I(1))==DISTANCE_BLOWUP
        %Then there was only one peak and it was bad, so ignore
        vote_value(c_idx) = 0;
        continue
    end
    
    %Get the pixel value of the peak nearest to the middle
    vote_value(c_idx) = img(x(I(1)), y(I(1)));
    
%     figure; %DEBUG
%     imagesc(img); hold on; %DEBUG
%     plot(y,x,'r+'); hold off; %DEBUG
%     title(num2str(c_idx)); %DEBUG
%     axis off; colormap gray; %DEBUG
    
end
%sort the vote values, to rerun both the max value and the ratio of the
%first pick and the second pic

[~,I] = sort(vote_value,'descend');
chanmax = I(1);
max_val = vote_value(I(1));
max_2nd_val = vote_value(I(2));

if max_2nd_val == 0
    max_2nd_val = 1;
end

confidence = max_val/max_2nd_val;


end

