
[mini, full, lens] = test_img_size();
save('conv_data_8th.mat', 'mini', 'full')

%if ~exist('conv_data.mat')
    %[mini, full, lens] = test_img_size();
    %save('conv_data.mat', 'mini', 'full')
%else
    %load conv_data.mat
%end

loadExperimentParams;
filter_size = params.SCALE_PYRAMID;
lw = 2;

plot(filter_size, mini.single.fft, 'b', 'DisplayName', 'single fft', 'LineWidth', lw)
hold on;
plot(filter_size, mini.single.fft_gpu, 'r', 'DisplayName', 'single fft gpu', 'LineWidth', lw)
%plot(filter_size, mini.single.imf, 'DisplayName', 'single imf', 'LineWidth', lw)
plot(filter_size, mini.single.imf_gpu, 'g', 'DisplayName', 'single imf gpu', 'LineWidth', lw)
hold off
legend('Location', 'northwest');
tname = sprintf('Single precision %d X %d X %d', lens);
title(tname)
ylabel('Time (s)')
xlabel('Filter size')
%saveas(gcf, tname);

figure
plot(filter_size, mini.double.fft, 'b', 'DisplayName', 'double fft', 'LineWidth', lw)
hold on;
plot(filter_size, mini.double.fft_gpu, 'r', 'DisplayName', 'double fft gpu', 'LineWidth', lw)
%plot(filter_size, mini.double.imf, 'DisplayName', 'double imf', 'LineWidth', lw)
plot(filter_size, mini.double.imf_gpu, 'g', 'DisplayName', 'double imf gpu', 'LineWidth', lw)
hold off
legend('Location', 'northwest');
tname = sprintf('Double precision %d X %d X %d', lens);
title(tname)
ylabel('Time (s)')
xlabel('Filter size')
%saveas(gcf, tname);

figure
plot(filter_size, mini.single.fft, 'b', 'DisplayName', 'single fft', 'LineWidth', lw)
hold on;
plot(filter_size, mini.single.fft_gpu, 'r', 'DisplayName', 'single fft gpu', 'LineWidth', lw)
%plot(filter_size, mini.single.imf, 'DisplayName', 'single imf', 'LineWidth', lw)
plot(filter_size, mini.single.imf_gpu, 'g', 'DisplayName', 'single imf gpu', 'LineWidth', lw)
plot(filter_size, mini.double.fft, 'b', 'LineStyle', '--', 'DisplayName', 'double fft', 'LineWidth', lw)
plot(filter_size, mini.double.fft_gpu, 'r', 'LineStyle', '--', 'DisplayName', 'double fft gpu', 'LineWidth', lw)
%plot(filter_size, mini.double.imf, 'DisplayName', 'double imf', 'LineWidth', lw)
plot(filter_size, mini.double.imf_gpu, 'g', 'LineStyle', '--', 'DisplayName', 'double imf gpu', 'LineWidth', lw)
hold off
legend('Location', 'northwest');
tname = sprintf('Convolve %d X %d X %d', lens);
title(tname)
ylabel('Time (s)')
xlabel('Filter size')
%saveas(gcf, tname);

figure
plot(filter_size, full.single.fft, 'b', 'DisplayName', 'single fft', 'LineWidth', lw)
hold on;
plot(filter_size, full.single.fft_gpu, 'r', 'DisplayName', 'single fft gpu', 'LineWidth', lw)
%plot(filter_size, full.single.imf, 'DisplayName', 'single imf', 'LineWidth', lw)
plot(filter_size, full.single.imf_gpu, 'g', 'DisplayName', 'single imf gpu', 'LineWidth', lw)
plot(filter_size, full.double.fft, 'b', 'LineStyle', '--', 'DisplayName', 'double fft', 'LineWidth', lw)
plot(filter_size, full.double.fft_gpu, 'r', 'LineStyle', '--', 'DisplayName', 'double fft gpu', 'LineWidth', lw)
%plot(filter_size, full.double.imf, 'DisplayName', 'double imf', 'LineWidth', lw)
plot(filter_size, full.double.imf_gpu, 'g', 'LineStyle', '--', 'DisplayName', 'double imf gpu', 'LineWidth', lw)
hold off
legend('Location', 'southeast');
tname = 'Convolve 2048 X 2048 X 141';
title(tname)
ylabel('Time (s)')
xlabel('Filter size')
%saveas(gcf, tname);


compute_err = @(X, ref) sum(sum(sum(abs(X - ref)))) / sum(ref(:));

function [mini, full, lens] = test_img_size()

    mini = {}; full = {}; 
    fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
    img = load3DTif_uint16(fn);
    %lens = floor(size(img) / 3);
    lens = floor(size(img) ./ [2, 4, 1]);
    img_mini = img(1:lens(1), 1:lens(2), :);
    tic;
    result = convn_cuda(single(img_mini), fspecial3('gaussian'));
    toc;

    %[double_times, single_times] = test_convn_dtype(img_mini);
    %mini.double = double_times;
    %mini.single = single_times;

    %full = struct;

    %[double_times, single_times] = test_convn_dtype(img);
    %full.double = double_times;
    %full.single = single_times;
end

function [double_times, single_times] = test_convn_dtype(img)
    i_size = size(img);
    fprintf('\n\nTesting with size: %d, %d, %d\n', i_size)
    double_times = test_convn_fsize(img);
    tic; fprintf('\n\nConvert to SINGLE / FLOAT precision\n\n'); img = single(img); toc;
    single_times = test_convn_fsize(img);
end

function [times] = test_convn_fsize(img)
    loadExperimentParams;
    filter_sizes = params.SCALE_PYRAMID;

    fft_times = [];
    fft_pad_times = [];
    fft_times_gpu = [];
    imf_times = [];
    imf_times_gpu = [];

    for fsize = filter_sizes
        fprintf('\nFilter size: %d\n', fsize);
        h = fspecial3('gaussian', fsize);
        [t_fft, t_fft_pad, t_fft_gpu, t_imf, t_imf_gpu] = test_convn_vers(img, h);
        fft_times = [fft_times t_fft];
        fft_pad_times = [fft_pad_times t_fft_pad];
        fft_times_gpu = [fft_times_gpu t_fft_gpu];
        imf_times = [imf_times t_imf];
        imf_times_gpu = [imf_times_gpu t_imf_gpu];
    end

    times = struct;
    times.fft = fft_times;
    times.fft_pad = fft_pad_times;
    times.imf = imf_times;
    times.fft_gpu = fft_times_gpu;
    times.imf_gpu = imf_times_gpu;
end

function [t_fft, t_fft_pad, t_fft_gpu, t_imf, t_imf_gpu] = test_convn_vers(img, h)

    %fprintf('Type: %s\n', class(img_mini));
    t_fft = 0;
    t_fft_pad = 0;
    t_fft_gpu = 0;
    t_imf = 0;
    t_imf_gpu = 0;

    options = {};
    options.Power2Flag = false;
    tic; 
    %timeout_len = 60;
    %fsize = size(h);
    %profile on -history; 
    img_blur_fft = convnfft(img, h, 'same', [], options); 
    %profile off; 
    %profsave(profile('info'), sprintf('pr-convnfft-%s-fsize%d', class(img), fsize(1))); 
    t_fft = toc;
    fprintf('`convnfft` power2flag false %s: %.4f\n', class(img), t_fft)

    %options.Power2Flag = true;
    %tic; 
    %img_blur_fft_pad = convnfft(img, h, 'same', [], options); 
    %t_fft_pad = toc;
    %fprintf('`convnfft` power2flag true %s: %.4f\n', class(img), t_fft_pad)

    options.GPU = true;
    gpuDevice(1); % reset GPU avail mem
    %profile on -history; 
    try
        tic; 
        img_blur_fft = convnfft(img, h, 'same', [], options); 
        t_fft_gpu = toc;
        fprintf('`convnfft` gpuArray power2flag false %s: %.4f\n', class(img), t_fft_gpu)
    catch
        void = toc;
        t_fft_gpu = 0;
        disp('GPU `convnfft` failed, out of memory')
    end
    %profile off; 
    %profsave(profile('info'), sprintf('pr-convnfft-gpu-%s-fsize%d', class(img), fsize(1))); 

    gpuDevice(1); % reset GPU avail mem
    tic; 
    try
        img_filter_func = convn_filter(img, h); 
        t_imf_gpu = toc;
        fprintf('`imfilter` gpuArray %s: %.4f\n', class(img), t_imf_gpu);
    catch
        void = toc;
        disp('GPU `convn_filter` failed, out of memory')
        %disp('GPU `convn_filter` failed, out of memory. Timing CPU:')
        %tic;
        %try
            %timeout('imfilter(img, h, ''same'', ''conv'')', timeout_len)
            %t_imf = toc;
        %catch
            %void = toc;
            %fprintf('`imfilter` exceeded timeout length: %d', timeout_len)
            %t_imf = 0;
        %end
        %fprintf('`imfilter` cpu %s: %.4f\n', class(img), t_imf);
    end
    
    %tic;
    %imfilter(img, h, 'same', 'conv');
    %t_imf = toc;
    %fprintf('`imfilter` cpu %s: %.4f\n', class(img), t_imf);

end


%%disp('convn cuda implementation')
%%tic; img_blur_cuda = convn_cuda(img, h); toc;
%%%compute_err(img_blur_fft, img_blur)

%%disp('array fire')
%%tic; img_blur_af = convn_arrayfire_cuda(img, h); toc;

%fprintf('convn MATLAB built-in (original) %s\n', class(img_mini))
%tic; img_blur_orig = convn(img_mini, h); toc;

%%fprintf('convn_filter with random filter func\n');
%%tic; i = convn_filter(img_mini, rand(filter_size, filter_size, filter_size)); toc;
