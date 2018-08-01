% Matlab claims that function arguments are passed by value such that local 
% changes to a variable do not affect the variable in the higher scope.
% it also claims to optimize to only pass by value for parts of variables that are being changed
% to test, the time of passing a large array should be equal to the time to pass a small array
% however upon testing large array is about 60% slower, tested over 1.5e+8 calls

a = ones(2048, 2048, 141);
b = ones(1,1,1);

runs = 150000000;

f1 = @() time_large(b, runs);

f2 = @() time_large2(a, runs);

timeit(f1)
timeit(f2)

function time_large(arr, runs)
    for i=1:runs
        large_arg(arr)
    end
end

function time_large2(arr, runs)
    for i=1:runs
        large_arg(arr)
    end
end
