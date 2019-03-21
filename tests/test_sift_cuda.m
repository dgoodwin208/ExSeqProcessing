function keys = test_sift_cuda(mode)

    % simple 1
%    img = zeros(3,3,3);
%    img(2,2,2) = 1;
%    img(1,2,2) = 0.5;
%    img(3,2,2) = 0.5;
%    img(2,1,2) = 0.5;
%    img(2,3,2) = 0.5;
%
%    kpts = [2,2,2];

    % simple 2
%    img = zeros(10,10,10);
%    img(5,5,5) = 1;
%    img(4,5,5) = 0.5;
%    img(6,5,5) = 0.5;
%    img(5,4,5) = 0.5;
%    img(5,6,5) = 0.5;
%
%    kpts = [5,5,5];

    % simple 3
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

    kpts = [3,3,3];

    % simple 4
    img=zeros(20,20,20);
    img(10,10,11) = 1;
    img(10,11,11) = 2;
    img(10,12,11) = 3;
    img(11,10,11) = 2;
    img(11,11,11) = 5; % keypoint
    img(11,12,11) = 1;
    img(12,10,11) = 1;
    img(12,11,11) = 3;
    img(12,12,11) = 2;

    img(10,10,10) = 0;
    img(10,11,10) = 1;
    img(10,12,10) = 2;
    img(11,10,10) = 1;
    img(11,11,10) = 3;
    img(11,12,10) = 0;
    img(12,10,10) = 0;
    img(12,11,10) = 2;
    img(12,12,10) = 1;

    kpts = [11,11,11];

    if strcmp(mode, 'mod')
        keys = calculate_3DSIFT_cuda(img,kpts,false);
    elseif strcmp(mode, 'org')
        keys = calculate_3DSIFT_org2(img,kpts,false);
    else
        fprintf('not support %s\n', mode);
        keys = [];
        return
    end

    disp('size(keys)')
    size(keys)

    for i=1:size(keys,1)
        fprintf('#### key{%d}.ivec\n',i);
        keys{i}.ivec
        fprintf('sum=%d\n',sum(keys{i}.ivec(:)));

        len = length(keys{i}.ivec(:));
        tmp = [(0:len-1)' double(keys{i}.ivec)']
        tmp(find(tmp(:,2) > 0),:)
    end

end

