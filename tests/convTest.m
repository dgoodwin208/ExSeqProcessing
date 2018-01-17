
fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
img = load3DTif_uint16(fn);
img_mini = img(:, :, :);
disp('Testing with size: ')

compute_err = @(X, ref) sum(sum(sum(abs(X - ref)))) / sum(ref(:));

h = fspecial3('gaussian', 20);

%disp('Built-in convn')
%if exist('img_blur.mat')
    %img_blur = loadmat('img_blur.mat');
    %% Takes ~6100 seconds (1.7 hrs) to create
    % Several minutes to load from file
%else
    %tic; img_blur = convn(img_mini, h, 'same'); toc;
%end

%disp('convnfft')
%tic; img_blur_fft = convnfft(img_mini, h, 'same'); toc;
%compute_err(img_blur_fft, img_blur)

disp('img size');
size(img_mini)
disp('filter size');
size(h)
disp('convn cuda implementation')
tic; img_blur_cuda = convn_cuda(img_mini, h); toc;
%compute_err(img_blur_fft, img_blur)

disp('convnfft power2flag false')
options = {};
options.Power2Flag = false;
tic; img_blur_fft = convnfft(img_mini, h, 'same', [], options); toc;
%compute_err(img_blur_fft, img_blur)


% Test with the batch size convolve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lens = floor(size(img) / 3);
img_mini = img(1:lens(1), 1:lens(2), 1:lens(3));

disp('img size');
size(img_mini)
disp('filter size');
size(h)
disp('convn cuda implementation')
tic; img_blur_cuda = convn_cuda(img_mini, h); toc;
%compute_err(img_blur_fft, img_blur)

disp('convnfft power2flag false')
options = {};
options.Power2Flag = false;
tic; img_blur_fft = convnfft(img_mini, h, 'same', [], options); toc;

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

