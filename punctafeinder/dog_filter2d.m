function img_dog = dog_filter2d(img)
FILTER_SIZE = 6;
STD1 = 4; 
STD2 = 5; 
gaussian1 = fspecial('Gaussian', FILTER_SIZE, STD1);
gaussian2 = fspecial('Gaussian', FILTER_SIZE, STD2);
dog = gaussian1 - gaussian2;
img_dog = conv2(double(img), dog, 'same');
