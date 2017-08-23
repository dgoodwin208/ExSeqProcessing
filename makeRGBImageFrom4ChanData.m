function [ rgbimg ] = makeRGBImageFrom4ChanData(data4d  )
%MAKERGBIMAGEFROM4CHANDATA Take an (X,Y,C) dataset and turn it into a 3D
%image in the RGB format. Tailored the colors to 
% Chan 1 = Blue
% Chan 2 = Green
% Chan 3 = Magenta
% Chan 4 = Red
AB = imfuse(data4d(:,:,1),data4d(:,:,4),'ColorChannels',[2 0 1]);
CD = imfuse(data4d(:,:,2),data4d(:,:,3),'ColorChannels',[2 1 2]);
rgbimg=AB+CD;

end


