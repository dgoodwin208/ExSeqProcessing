function [ outputImg ] = makeDebugImageOfPoints(points,imgSize)


%makeDebugImageOfPoints Summary of this function goes here
%   Detailed explanation goes here
outputImg = zeros(imgSize);

for i = 1:size(points,1)
   
    centroid_pos = round(points(i,:));
    
    %Watch out for the XY shift
    y_indices = (centroid_pos(1) - 1):(centroid_pos(1) + 1);
    y_indices(y_indices<1)=[];y_indices(y_indices>imgSize(1))=[];
    
    x_indices = (centroid_pos(2) - 1):(centroid_pos(2) + 1);
    x_indices(x_indices<1)=[];x_indices(x_indices>imgSize(2))=[];
    
    z_indices = (centroid_pos(3) - 1):(centroid_pos(3) + 1);
    z_indices(z_indices<1)=[];z_indices(z_indices>imgSize(3))=[];
    
    outputImg(y_indices,x_indices,z_indices) = 100;

end


end

