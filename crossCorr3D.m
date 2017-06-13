function [ xcorr_scores ] = crossCorr3D(data1,data2,offsets )
%CROSSCORR3D Summary of this function goes here
%   Detailed explanation goes here

border_mask = zeros(size(data1));
border_mask(1:offsets(1)+1,:,:)=1; border_mask(end-offsets(1):end,:,:)=1;
border_mask(:,1:offsets(2)+1,:)=1; border_mask(:,end-offsets(2):end,:)=1;
border_mask(:,:,1:offsets(3)+1)=1; border_mask(:,:,end-offsets(3):end)=1;
border_mask = logical(border_mask);
%%
xcorr_scores = zeros(offsets);
for z = -1*offsets(3):offsets(3)
    for y = -1*offsets(2):offsets(2)
        for x = -1*offsets(1):offsets(1)
            data2_shift = circshift(data2,x,1);
            data2_shift = circshift(data2_shift,y,2);
            data2_shift = circshift(data2_shift,z,3);
            
            
            %No matter what the shift, zero out any border that could
            %bias the result as a functino of shift
            data2_shift(border_mask) = 0.;
            
            xcorr_scores(x+offsets(1)+1,y+offsets(2)+1,z+offsets(3)+1) = ...
                sum(data1(:).*data2_shift(:));
            
            %fprintf('Calculated z=%i y=%i x=%i: %f\n',z,y,x,xcorr_scores(x+offsets(1)+1,y+offsets(2)+1,z+offsets(3)+1));
        end
    end
end

end


