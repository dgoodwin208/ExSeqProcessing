
%%normalized gaussian kernel
%a=[.0003 .1065 .7866 .1065 .0003]; 
%n=length(a);

%%make a 4D kernel

%b=a;
%for d=2:4
    %b=kron(b,a');
%end
%H=reshape(b,n*ones(1,4));

H = fspecial3('gaussian');
%H = [1 0 -1; 2 0 -2; 1 0 -1];
%a = H(1, :);
n=length(a);

%make a test matrix
%A = randn(2048,2048,141);
A = randn(10,10,10);

B = convn(A, H, 'same');
original_size = size(B);

%%compare execution times
%sprintf('convnfft')
%options = {};
%options.Power2Flag = false;
%tic
%B = convnfft(H,A,'same', [], options);
%toc

sprintf('convnsep')
tic
B2 = convnsep({a,a,a},A, 'same');
toc
b2_size = size(B2);
assert(isequal(original_size, b2_size));

sprintf('convnsep custom')
tic
B3 = conv3_sep(A, a);
toc

tic
B4 = convn(A, a, 'same');

%compare error to size of the result of convn
norm(B(:)-B2(:))
norm(B(:))
%(abs(norm(B(:)-B2(:))))/ norm(B(:))
%(abs(norm(B(:)-B3(:))))/ norm(B(:))
%B(:, :, 1)
%B3(:, :, 1)

%B
%B2
