function tests = mex_meanfilesTest
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

    for i = 1:4
        fid = fopen(fullfile(testCase.TestData.tempDir,sprintf('test_%d.bin',i)),'w');
        fwrite(fid,data{i}','double');
        fclose(fid);
    end
    testCase.TestData.data = data;

    testCase.TestData.meanfile_list = {'test_1.bin','test_2.bin','test_3.bin','test_4.bin'};
end

function teardown(testCase)
    rmdir(testCase.TestData.tempDir,'s');
end

% normal test cases
function testMeanFiles(testCase)
    mean_org = (testCase.TestData.data{1}+testCase.TestData.data{2}+testCase.TestData.data{3}+testCase.TestData.data{4})/4;

    mean_file = 'test_mean.bin';
    meanfiles(testCase.TestData.tempDir,testCase.TestData.meanfile_list,mean_file);

    fid = fopen(fullfile(testCase.TestData.tempDir,mean_file),'r');
    mean_mod = fread(fid,[1,Inf],'double');
    fclose(fid);

    verifyEqual(testCase,mean_mod,mean_org(:,1)');
end

% error test cases

