function keys = test_sift_cuda_w_harris3d(mode)

    %img=zeros(100,100,50);
    %for i=0:3
    %    for j=0:3
    %        img(30+i,30+j,20)=5;
    %    end
    %end
%    img=zeros(20,20,20);
%    for i=0:1
%        for j=0:1
%            for k=0:1
%                img(3+i,3+j,3+k)=5;
%                img(7+i,7+j,7+k)=5;
%            end
%        end
%    end
%    img(5,1,1);

%    img=zeros(5,5,5);
%    img(3,3,3) = 1;
%    img(2,3,3) = 0.5;
%    img(4,3,3) = 0.5;
%    img(3,2,3) = 0.5;
%    img(3,4,3) = 0.5;

    img=zeros(5,5,5);
    img(2,2,3) = 1;
    img(2,3,3) = 2;
    img(2,4,3) = 3;
    img(3,2,3) = 2;
    img(3,3,3) = 5; % keypoint
    img(3,4,3) = 1;
    img(4,2,3) = 1;
    img(4,3,3) = 3;
    img(4,4,3) = 2;

    img(2,2,2) = 0;
    img(2,3,2) = 1;
    img(2,4,2) = 2;
    img(3,2,2) = 1;
    img(3,3,2) = 3;
    img(3,4,2) = 0;
    img(4,2,2) = 0;
    img(4,3,2) = 2;
    img(4,4,2) = 1;


    blur_size = 5;
    skipDescriptors = true;

    if strcmp(mode, 'mod')
        keys = run_mod(img, blur_size, skipDescriptors);
    elseif strcmp(mode, 'mod2')
        keys = run_mod2(img, blur_size, skipDescriptors);
    elseif strcmp(mode, 'org')
        keys = run_org(img, blur_size, skipDescriptors);
    else
        fprintf('not support %s\n', mode);
        keys = [];
        return
    end

    disp('size(keys)')
    size(keys)

    if skipDescriptors == false
        for i=1:size(keys,1)
            fprintf('#### key{%d}.ivec\n',i);
            keys{i}.ivec
            fprintf('sum=%d\n',sum(keys{i}.ivec(:)));
        end
    end

end

function keys = run_mod(img, blur_size, skipDescriptors)

    options = {};
    options.Power2Flag = false;
    options.CastSingle = true;

    res_vect = Harris3D(img,blur_size,options);
    %res_vect = Harris3D_mod(img,blur_size,options); % convnfft -> convn
    disp('size(res_vect)')
    size(res_vect)

    h = fspecial3('gaussian',blur_size);
    img_blur = convnfft(img,h,'same',[],options);

    keys = calculate_3DSIFT_cuda(img_blur,res_vect,skipDescriptors);

end

function keys = run_mod2(img, blur_size, skipDescriptors)

    options = {};
    options.Power2Flag = false; % not effective
    options.CastSingle = true; % not effective

    res_vect = Harris3D_mod(img,blur_size,options); % convnfft -> convn
    disp('size(res_vect)')
    size(res_vect)

    h = fspecial3('gaussian',blur_size);
    img_blur = convn(img,h,'same');

    keys = calculate_3DSIFT_cuda(img_blur,res_vect,skipDescriptors);

end

function keys = run_org(img, blur_size, skipDescriptors)

    res_vect = Harris3D_org(img,blur_size);
    disp('size(res_vect)')
    size(res_vect)

    h = fspecial3('gaussian',blur_size);
    img_blur = convn(img,h,'same');

    keys = calculate_3DSIFT_org2(img_blur,res_vect,skipDescriptors);

end
