function  out_img = save3DTif(imgvol, outputFileName,doNormalize)

    if nargin ==2
        doNormalize = 0;
    end
    %SAVE3DTif: Load 3D a tif into a y,x,z stack
    %fname has to end in .tif
    
    %If you don't trust the input values to be within 0-65535 you can use
    if doNormalize 
        %set the minimum to zero
        imgvol = imgvol - min(imgvol(:));
        %set the max to max uint16
        imgvol = (imgvol/max(imgvol(:)))*(2^16-1);
        imgvol = uint16(imgvol);
    else
        imgvol = uint16(imgvol);
    end
    
    if exist(outputFileName,'file')
        delete(outputFileName);
    end
    
   for K=1:size(imgvol,3)
    imwrite(imgvol(:, :, K), outputFileName, 'WriteMode', 'append',  'Compression','none');
   end
    
    
end

