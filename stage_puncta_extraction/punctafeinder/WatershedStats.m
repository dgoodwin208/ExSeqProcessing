for i = 1:20
    for j = 1:4
        
        numpixels = [];
        for k = 1:length(centroids{i,j})
            numpixels(k) = length(centroids{i,j}(k).PixelIdxList);
        end
        idx = find(numpixels>7);
        avg_numpixels(i,j) = mean(numpixels(idx));
    end
    numspots (i,j) = length(idx);
end
