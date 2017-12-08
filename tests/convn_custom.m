function A = convn_custom(A, B, power2flag)
% CONVN_CUSTOM  FFT-BASED N-dimensional convolution.
%   C = CONVN_CUSTOM(A, B) performs the N-dimensional convolution of
%   matrices A and B. If nak = size(A,k) and nbk = size(B,k), then
%   size(C,k) = max([nak+nbk-1,nak,nbk]);
%
% METHOD: CONVN_CUSTOM uses Fourier transform (FT) convolution theorem, i.e.
%         FT of the convolution is equal to the product of the FTs of the
%         input functions.
%         In 1-D, the complexity is O((na+nb)*log(na+nb)), where na/nb are
%         respectively the lengths of A and B.

if nargin < 3 || isempty(power2flag)
    power2flag = true;
end

if power2flag
    % faster FFT if the dimension is power of 2
    trunc_func = @(m, n) 2^nextpow2(m + n - 1);
else
    % slower, but smaller temporary arrays
    trunc_func = @(m, n) m + n - 1; 
end
ifun = @(m,n) ceil((n-1)/2)+(1:m);

ABreal = isreal(A) && isreal(B);

truncs = [];
subs(1:ndims(A)) = {':'};
for i=1:ndims(A)
    m = size(A, i);
    n = size(B, i);
    truncs = [truncs trunc_func(m, n)];
    subs{i} = ifun(m, n);
end
%truncs(3) = 2^nextpow2(size(A,3));

disp(['fftn'])
tic;
B = fftn(B, truncs);
A = fftn(A, truncs);
toc;

disp(['fft'])
tic;
for dim=1:ndims(A)
    % compute the FFT length
    A = fft(A,truncs(dim),dim);
    B = fft(B,truncs(dim),dim);
end
toc;

disp(['fft orig dimension'])
truncs = [2048, 2048, 141]
tic;
for dim=1:ndims(A)
    % compute the FFT length
    A = fft(A,truncs(dim),dim);
    B = fft(B,truncs(dim),dim);
end
toc;


A = A.*B;
clear B;

disp(['ifft'])
tic;
for dim=1:ndims(A)
    A = ifft(A,[],dim);
end
toc;

disp(['ifftn'])
tic;
A = ifftn(A);
toc;

if ABreal
    A = real(A(subs{:}));
end

end
