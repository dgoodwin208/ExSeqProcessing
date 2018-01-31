
fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
img = load3DTif_uint16(fn);

compute_err = @(X, ref) sum(sum(sum(abs(X - ref)))) / sum(ref(:));

filter_size = 10
h = fspecial3('gaussian', filter_size);
%tic; disp('convert to single'); img = single(img); h = single(h); toc;

%tic; disp('convert to single'); img = single(img); h = single(h); toc;

%disp('Built-in convn')
%if exist('img_blur.mat')
    %img_blur = loadmat('img_blur.mat');
    %% Takes ~6100 seconds (1.7 hrs) to create
    % Several minutes to load from file
%else
    %tic; img_blur = convn(img, h, 'same'); toc;
%end

%disp('convnfft')
%tic; img_blur_fft = convnfft(img, h, 'same'); toc;
%compute_err(img_blur_fft, img_blur)

%disp('convn cuda implementation')
%tic; img_blur_cuda = convn_cuda(img, h); toc;
%%compute_err(img_blur_fft, img_blur)

%disp('Testing with size: ')
%disp('img size'); size(img)
%disp('filter size'); size(h)

%disp('convnfft power2flag false single')
%options = {};
%options.Power2Flag = false;
%tic; img_blur_fft = convnfft(img, h, 'same', [], options); toc;

%%test imfilter Fastest implementation yet
%disp('imfilter gpuArray single');
%tic;
%gpuDevice(1); %free up max memory
%img_gpu = gpuArray(img);
%h_gpu = gpuArray(h);
%img_imfilter = imfilter(img_gpu, h_gpu, 'same', 'conv');
%toc;

%disp('convnfft power2flag false single')
%options = {};
%options.Power2Flag = false;
%tic; img_blur_fft = convnfft(img, h, 'same', [], options); toc;

%disp('array fire')
%tic; img_blur_af = convn_arrayfire_cuda(img, h); toc;

% Test with the batch size convolve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lens = floor(size(img) / 3);
img_mini = img(1:lens(1), 1:lens(2), :);

disp('Testing with size: ')
disp('img size'); size(img_mini)
disp('filter size'); size(h)
fprintf('Type: %s\n', class(img_mini));

%fprintf('convnfft power2flag false, %s\n', class(img_mini))
%options = {};
%options.Power2Flag = false;
%tic; img_blur_fft = convnfft(img_mini, h, 'same', [], options); toc;

%fprintf('imfilter gpuArray, %s\n', class(img_mini));
%tic;
%img_gpu = gpuArray(img_mini);
%h_gpu = gpuArray(h);
%img_imfilter = imfilter(img_gpu, h_gpu, 'same', 'conv');
%toc;

tic; disp('convert to single'); img_mini = single(img_mini); toc;

%disp('Testing with size: ')
%disp('img size'); size(img_mini)
%disp('filter size'); size(h)
%fprintf('Type: %s\n', class(img_mini));

fprintf('convnfft power2flag false, %s\n', class(img_mini))
options = {};
options.Power2Flag = false;
tic; img_blur_fft = convnfft(img_mini, h, 'same', [], options); toc;

fprintf('imfilter gpuArray, %s ', class(img_mini));
fprintf('convn_filter func\n');
tic; img_filter_func = convn_filter(img_mini, h); toc;

%fprintf('convn_filter with random filter func\n');
%tic; i = convn_filter(img_mini, rand(filter_size, filter_size, filter_size)); toc;


%disp('convn cuda implementation')
%tic; img_blur_cuda = convn_cuda(img_mini, h); toc;
%size(img_blur_cuda);
%compute_err(img_blur_fft, img_blur_cuda)

%compute_err(img_blur_fft, img_blur)
%disp('Custom FFT based matlab implementation')
%tic; img_blur_cust = convn_custom(img_mini, h, false); toc;
%compute_err(img_blur_cust, img_blur_fft)

%disp('Custom FFT based matlab implementation power2flag')
%tic; img_blur_cust_pow2 = convn_custom(img_mini, h, true); toc;
%compute_err(img_blur_cust_pow2, img_blur)

%disp('Custom FFT based CUDA implementation')
%tic;
%toc;
%sprintf('\n\n')

