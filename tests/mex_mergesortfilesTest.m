function tests = mex_mergesortfilesTest
    tests = functiontests(localfunctions);
end

function setup(testCase)
    testCase.TestData.tempDir = '/mp/nvme0/tmp/testworks';
    if ~exist(testCase.TestData.tempDir)
        mkdir(testCase.TestData.tempDir)
    end

    n = 4000;
    data = {};
    for i = 1:4
        data{i}=[rand(n,1) (1:n)'];
    end
    data{1}(2,1)=0.1;
    data{1}(4,1)=0.1;
    for i = 1:4
        data{i}=sortrows(data{i});
    end

    for i = 1:4
        fid = fopen(fullfile(testCase.TestData.tempDir,sprintf('test_%d.bin',i)),'w');
        fwrite(fid,data{i}','double');
        fclose(fid);
    end
    testCase.TestData.data = data;

    testCase.TestData.mergefile_list = {};
    testCase.TestData.mergefile_list{1} = { 'test_1.bin','test_2.bin','test_1-2.bin' };
    testCase.TestData.mergefile_list{2} = { 'test_3.bin','test_4.bin','test_3-4.bin' };
    testCase.TestData.mergefile_list{3} = { 'test_1-2.bin','test_3-4.bin','test_1-4.bin' };
end

function teardown(testCase)
    rmdir(testCase.TestData.tempDir,'s');
end

% normal test cases
function testMergeSort(testCase)
    sorted = sortrows([testCase.TestData.data{1}; testCase.TestData.data{2}; testCase.TestData.data{3}; testCase.TestData.data{4}]);

    mergesortfiles(testCase.TestData.tempDir,testCase.TestData.mergefile_list);
    fid = fopen(fullfile(testCase.TestData.tempDir,'test_1-4.bin'),'r');
    mergesorted = fread(fid,[2,Inf],'double');

    verifyEqual(testCase,mergesorted,sorted');
end

% error test cases

