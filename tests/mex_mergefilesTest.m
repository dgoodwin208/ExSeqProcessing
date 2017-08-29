function tests = mex_mergefilesTest
    tests = functiontests(localfunctions);
end

function setup(testCase)
    if ~exist('/mp/nvme0/tmp/testworks')
        mkdir('/mp/nvme0/tmp/testworks')
    end

    n = 4000;
    data = {};
    for i = 1:4
        data{i}=[rand(n,1); (1:n)'];
    end
    data{1}(2,1)=0.1;
    data{1}(4,1)=0.1;

    for i = 1:4
        fid = fopen(fullfile('/mp/nvme0/tmp/testworks',sprintf('test_%d.bin',i)),'w');
        fwrite(fid,data{i},'double');
        fclose(fid);
    end

    testCase.mergefile_list = {};
    testCase.mergefile_list{1} = { '/mp/nvme0/tmp/testworks/test_1.bin','/mp/nvme0/tmp/testworks/test_2.bin','/mp/nvme0/tmp/testworks/test_1-2.bin' };
    testCase.mergefile_list{2} = { '/mp/nvme0/tmp/testworks/test_3.bin','/mp/nvme0/tmp/testworks/test_4.bin','/mp/nvme0/tmp/testworks/test_3-4.bin' };
    testCase.mergefile_list{3} = { '/mp/nvme0/tmp/testworks/test_1-2.bin','/mp/nvme0/tmp/testworks/test_3-4.bin','/mp/nvme0/tmp/testworks/test_1-4.bin' };
end

function teardown(testCase)
    rmdir('/mp/nvme0/tmp/testworks','s');
end

% normal test cases
function testMerge(testCase)
    mergefiles('./testworks',

end

% error test cases

