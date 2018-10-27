

filedir = 'C:\Users\chenf\Desktop\Output\Renamed\';
%filedir = '/Users/Shirin/Desktop/color/';

% File name convenction exframe7upscaled_round001_ch01
FILEROOT_NAME_INPUT = 'exseqautoframe7i'


parpool(10)

parfor round_num = 1:20
    
    filepath_split1 = fullfile(fi,sprintf('%s_round%.03i',FILEROOT_NAME_INPUT,round_num))
    filepath_split2 = '_registered.tif';
    
    for chan_num = 1:4
        tic
        if chan_num == 1
           chan = sprintf('_ch0%i',chan_num-1)
        else
            chan = sprintf('_ch0%iSHIFT',chan_num-1);
        end
        stack_in = uint16(load3DTif([filepath_split1 chan filepath_split2]));
        background = min(stack_in,[],3);
        toc
        tic
        stack_in = dog_filter(stack_in); 
  
        
        back_dog = dog_filter2d(background); back_dogmax = max(back_dog(:)); 
        fgnd_mask = zeros(size(stack_in)); 
        fgnd_mask(stack_in>back_dogmax) = 1; % use first slice to determine threshold for dog
        fgnd_mask = logical(fgnd_mask); % and get mask
        
        stack_in(~fgnd_mask) = 0; % thresholded using dog background

        neg_masked_image = -int32(stack_in); 
        neg_masked_image(~stack_in) = inf; 
        toc
        tic
        L = watershed(neg_masked_image);
        L(~stack_in) = 0;
        fprintf('wshed\n'); 
        %watershed_out{chan_num} = L;     
        toc
        stats = regionprops(L, 'PixelIdxList', 'Area');
        centroids{round_num, chan_num} = regionprops(L, 'Centroid', 'PixelIdxList');
                
        % need to move this up before the save. 
        %eliminate spots with 6 or less voxels
        for i= 1:length(stats)
            if stats(i).Area < 7
                L(stats(i).PixelIdxList) = 0;
            end
        end
        
        save_img(L, [filepath_split1 chan '_L' filepath_split2]);  
    end
 
end

