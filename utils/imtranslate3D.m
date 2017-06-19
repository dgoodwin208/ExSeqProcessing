function [ data_shifted] = imtranslate3D(data,shifts )
%SHIFT3D Works like imtranslate but for 3 dimensions


%Use the circular shift feature in matlab, which wraps elements around from
%the end to positition 1, etc. Then we have to go back to zero out
data_shifted = circshift(data,shifts);

if shifts(1)<0
    %note there is a tricky addition of 1 here due to funkiness with 'end'
    data_shifted(end+shifts(1)+1:end,:,:) = 0.;
elseif shifts(1)>0
    data_shifted(1:shifts(1),:,:) = 0.;
end

if shifts(2)<0
    %note there is a tricky addition of 1 here due to funkiness with 'end'
    data_shifted(:,end+shifts(2)+1:end,:) = 0.;
elseif shifts(2)>0
    data_shifted(:,1:shifts(2),:) = 0.;
end

if shifts(3)<0
    %note there is a tricky addition of 1 here due to funkiness with 'end'
    data_shifted(:,:,end+shifts(3)+1:end) = 0.;
elseif shifts(3)>0
    data_shifted(:,:,1:shifts(3)) = 0.;
end


end


