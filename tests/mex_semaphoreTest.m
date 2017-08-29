function tests = mex_semaphoreTest
    tests = functiontests(localfunctions);
end

function setup(testCase)
    semaphore('/t','unlink');
end

function teardown(testCase)
    semaphore('/t','unlink');
end

% normal test cases
function testOpen(testCase)
    ret = semaphore('/t','open',3);

    verifyEqual(testCase,ret,0);
end

function testUnlink(testCase)
    ret = semaphore('/t','open',3);
    ret = semaphore('/t','unlink');

    verifyEqual(testCase,ret,0);
end

function testGetValue(testCase)
    ret = semaphore('/t','open',3);
    val = semaphore('/t','getvalue');

    verifyEqual(testCase,val,3);
end

function testWaitAndPost(testCase)
    ret = semaphore('/t','open',3);
    ret = semaphore('/t','wait');

    verifyEqual(testCase,ret,0);

    val = semaphore('/t','getvalue');

    verifyEqual(testCase,val,2);

    ret = semaphore('/t','post');

    verifyEqual(testCase,ret,0);

    val = semaphore('/t','getvalue');

    verifyEqual(testCase,val,3);
end

function testTrywait(testCase)
    ret = semaphore('/t','open',3);
    ret = semaphore('/t','trywait');

    verifyEqual(testCase,ret,0);
    val = semaphore('/t','getvalue');

    verifyEqual(testCase,val,2);
end

function testExclusive(testCase)
    parpool(3);
    ret = semaphore('/t','open',2);
    
    parfor i = 1:3
        pause(i*2);
        ret = semaphore('/t','trywait');

        if i == 1 || i == 2
            verifyEqual(testCase,ret,0);
        else 
            verifyEqual(testCase,ret,-1);
            pause(2);

            ret = semaphore('/t','trywait');
            verifyEqual(testCase,ret,0);
        end
    end
end

% error test cases

function testLessArgs(testCase)
    ret = semaphore('/t');

    verifyNotEqual(testCase,ret,0);
end

function testNotOpenAndTrywaitError(testCase)
    ret = semaphore('/t','trywait');

    verifyNotEqual(testCase,ret,0);
end

function testTrywaitError(testCase)
    ret = semaphore('/t','open',1);
    ret = semaphore('/t','trywait');

    verifyEqual(testCase,ret,0);

    ret = semaphore('/t','trywait');

    verifyEqual(testCase,ret,-1);

    val = semaphore('/t','getvalue');

    verifyEqual(testCase,val,0);
end

