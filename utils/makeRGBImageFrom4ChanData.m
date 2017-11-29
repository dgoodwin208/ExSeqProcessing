function [ rgbimg ] = makeRGBImageFrom4ChanData(data4d,clims)
%MAKERGBIMAGEFROM4CHANDATA Take an (X,Y,C) dataset and turn it into a 3D
%image in the RGB format. Tailored the colors to
% Chan 1 = Blue
% Chan 2 = Green
% Chan 3 = Magenta
% Chan 4 = Red
%NOTE! The data4d input MUST be between 0 and 1.

if nargin==1
    
    AB = imfuse(data4d(:,:,1),data4d(:,:,4),'ColorChannels',[2 0 1]);
    CD = imfuse(data4d(:,:,2),data4d(:,:,3),'ColorChannels',[2 1 2]);
    rgbimg=AB+CD;
else
%     r = zeros(size(data4d,1),size(data4d,2));
%     g = zeros(size(data4d,1),size(data4d,2));
%     b = zeros(size(data4d,1),size(data4d,1));
    
    mapped_chan1 = mymap(data4d(:,:,1),clims(1,:));
    mapped_chan2 = mymap(data4d(:,:,2),clims(2,:));
    mapped_chan3 = mymap(data4d(:,:,3),clims(3,:));
    mapped_chan4 = mymap(data4d(:,:,4),clims(4,:));
    
    %Red channel is the sum of Chan 4 (red) and Chan 3(Magenta) 
    r= uint8(mapped_chan4 + mapped_chan3); 
    
    %Blue chan is the sum of Chan1 (Blue) and Chan 3(Magenta)
    b= uint8(mapped_chan1 + mapped_chan3); 
    
    %Green channel is just Chan2
    g= uint8(mapped_chan2); 
    
    rgbimg = zeros(size(r,1),size(r,2),3);
    rgbimg(:,:,1) = r;
    rgbimg(:,:,2) = g;
    rgbimg(:,:,3) = b;
end


%
function mapped_val = mymap(inputImg,clims)
    minVal = clims(1);
    maxVal = clims(2);
    mapped_val = uint8(zeros(size(inputImg)));
    
    indices_Min = inputImg<=minVal;
    indices_Max = inputImg>=maxVal;
    indicesRest = inputImg>minVal && inputImg<maxVal;
    
    mapped_val(indices_Min)=0;
    mapped_val(indices_Max)=255;
    mapped_val(indicesRest) = round(255*(inputImg(indicesRest)-minVal)/(maxVal-minVal));
    
end

end


