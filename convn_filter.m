function img_blur = convn_filter(img, h)

        img_gpu = gpuArray(img);
        h_gpu = gpuArray(h);
        img_blur = imfilter(img_gpu, h_gpu, 'same', 'conv');
        img_blur = double(img_blur);

end
