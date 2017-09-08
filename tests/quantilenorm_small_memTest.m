function tests = quantilenorm_small_memTest
    tests = functiontests(localfunctions);
end

function setup(testCase)
    quantilenorm_init([5]);

    testCase.TestData.tempDir = '/mp/nvme0/tmp/testworks';
    if ~exist(testCase.TestData.tempDir)
        mkdir(testCase.TestData.tempDir)
    end

    img={};
    for i = 1:4
        img{i}=rand(10,10,80);
    end
    img{1}(2,1,1)=0.1;
    img{1}(4,1,1)=0.1;
    for i = 1:4
        save3DTif(img{i},fullfile(testCase.TestData.tempDir,sprintf('test_%d.tif',i)));
    end

end

function teardown(testCase)
    quantilenorm_final(1);

    rmdir(testCase.TestData.tempDir,'s');
end

% utility functions
function image = load_binary_image(outputdir,image_fname,image_height,image_width)
    fid = fopen(fullfile(outputdir,image_fname),'r');
    count = 1;
    while ~feof(fid)
        sub_image = fread(fid,[2,image_height*image_width],'double');
        %sub_image = fread(fid,[image_height,image_width],'double');
        if ~isempty(sub_image)
            image(:,:,count) = reshape(sub_image(2,:),[image_height image_width]);
            %image(:,:,count) = sub_image;
            count = count + 1;
        end
    end
    fclose(fid);
end

% normal test cases
function testQuantilenorm(testCase)
    t1 = load3DTif(fullfile(testCase.TestData.tempDir,'test_1.tif'));
    t2 = load3DTif(fullfile(testCase.TestData.tempDir,'test_2.tif'));
    t3 = load3DTif(fullfile(testCase.TestData.tempDir,'test_3.tif'));
    t4 = load3DTif(fullfile(testCase.TestData.tempDir,'test_4.tif'));

    norm_org = quantilenorm([t1(:) t2(:) t3(:) t4(:)]);

    [norm1_fname,norm2_fname,norm3_fname,norm4_fname,image_height,image_width] = quantilenorm_small_mem( ...
        testCase.TestData.tempDir,'test', ...
        fullfile(testCase.TestData.tempDir,'test_1.tif'), ...
        fullfile(testCase.TestData.tempDir,'test_2.tif'), ...
        fullfile(testCase.TestData.tempDir,'test_3.tif'), ...
        fullfile(testCase.TestData.tempDir,'test_4.tif'));

    n1 = load_binary_image(testCase.TestData.tempDir,norm1_fname,image_height,image_width);
    n2 = load_binary_image(testCase.TestData.tempDir,norm2_fname,image_height,image_width);
    n3 = load_binary_image(testCase.TestData.tempDir,norm3_fname,image_height,image_width);
    n4 = load_binary_image(testCase.TestData.tempDir,norm4_fname,image_height,image_width);

    verifyEqual(testCase,n1(:),norm_org(:,1),'RelTol',1e-2);
    verifyEqual(testCase,n2(:),norm_org(:,2),'RelTol',1e-2);
    verifyEqual(testCase,n3(:),norm_org(:,3),'RelTol',1e-2);
    verifyEqual(testCase,n4(:),norm_org(:,4),'RelTol',1e-2);
end

