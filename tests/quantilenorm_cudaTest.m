function tests = quantilenorm_cudaTest
    tests = functiontests(localfunctions);
end

function setup(testCase)
    if ~exist('logs')
        mkdir('logs');
    end

    num_gpus = gpuDeviceCount();
    quantilenorm_cuda_init(ones(1,num_gpus),[num_gpus+2,2]);

    testCase.TestData.tempDir = '/mp/nvme0/tmp/testworks';
    if ~exist(testCase.TestData.tempDir)
        mkdir(testCase.TestData.tempDir)
    end

    img={};
    for i = 1:4
        img{i}=rand(10,10,80);
%        img{i}=rand(5,5,3);
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
function image = load_binary_image(outputdir,image_fname)
    fid = fopen(fullfile(outputdir,image_fname),'r');
    image = fread(fid,'double');
    fclose(fid);
end

% normal test cases
function testQuantilenorm(testCase)
    t1 = load3DTif(fullfile(testCase.TestData.tempDir,'test_1.tif'));
    t2 = load3DTif(fullfile(testCase.TestData.tempDir,'test_2.tif'));
    t3 = load3DTif(fullfile(testCase.TestData.tempDir,'test_3.tif'));
    t4 = load3DTif(fullfile(testCase.TestData.tempDir,'test_4.tif'));

    norm_org = quantilenorm([t1(:) t2(:) t3(:) t4(:)]);

%    [norm1_fname,norm2_fname,norm3_fname,norm4_fname,image_height,image_width] = quantilenorm_cuda( ...
    r = quantilenorm_cuda( ...
        testCase.TestData.tempDir,'test_result', { ...
        fullfile(testCase.TestData.tempDir,'test_1.tif'), ...
        fullfile(testCase.TestData.tempDir,'test_2.tif'), ...
        fullfile(testCase.TestData.tempDir,'test_3.tif'), ...
        fullfile(testCase.TestData.tempDir,'test_4.tif')});

    norm1_fname = r{1};
    norm2_fname = r{2};
    norm3_fname = r{3};
    norm4_fname = r{4};
    image_height = r{5};
    image_width  = r{6};

    n1 = load_binary_image(testCase.TestData.tempDir,norm1_fname);
    n2 = load_binary_image(testCase.TestData.tempDir,norm2_fname);
    n3 = load_binary_image(testCase.TestData.tempDir,norm3_fname);
    n4 = load_binary_image(testCase.TestData.tempDir,norm4_fname);

    threshold = 1e-2;
    verifyEqual(testCase,n1(:),norm_org(:,1),'RelTol',threshold);
    verifyEqual(testCase,n2(:),norm_org(:,2),'RelTol',threshold);
    verifyEqual(testCase,n3(:),norm_org(:,3),'RelTol',threshold);
    verifyEqual(testCase,n4(:),norm_org(:,4),'RelTol',threshold);
end

