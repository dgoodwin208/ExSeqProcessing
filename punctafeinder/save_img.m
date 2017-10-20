function [] = save_img(img, outpath)
     % save img
    img_ = uint16(img(:,:,1)); 
    imwrite(img_, outpath);
    for j = 2:size(img, 3) % aka #z slices
        img_ = uint16(img(:,:,j));
        imwrite(img_,outpath, 'WriteMode','append')
    end
    fprintf('img saved\n');
end
    