function [scores,puncta_size] = callRoundROI(puncta)


%Sum the four channels together
four_chans = sum(puncta,4);

level = multithresh(four_chans);
all_slices_bw = zeros(size(four_chans));
for z=1:size(four_chans,3)
    bw = imbinarize(four_chans(:,:,z),level);
    all_slices_bw(:,:,z) = bw;
end

D = bwdist(~all_slices_bw);
D = -D;
D(~all_slices_bw) = Inf;
L = watershed(D);
L(~all_slices_bw) = 0;


s = regionprops(L, four_chans, {'WeightedCentroid'});
subregioncenter = repmat([mean(1:10),mean(1:10),mean(1:10)],length(s),1);
region_centroids = zeros(length(s),3);
for s_idx = 1:length(s)
    region_centroids(s_idx,:) = s(s_idx).WeightedCentroid;
end
D = diag(pdist2(subregioncenter,region_centroids,'euclidean'));
[~,min_centroid_idx] = min(D);

mask = L==min_centroid_idx;
puncta_size = sum(mask(:));
% figure;
% for z = 1:10
%     subplot(1,10,z)
%     imagesc(mask(:,:,z),[0 2])
%     axis off;
% end
% figure;

scores = [];
for c_idx = 1:4
    puncta_img = puncta(:,:,:,c_idx).*mask;
    scores(c_idx) = sum(puncta_img(:));
    
end

end
