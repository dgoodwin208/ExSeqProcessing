
%shortcut to save matlab workspace memory
test_img_size();

%actual
%[mini, full, lens, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_img_size();
%save('conv_data_8th.mat', 'mini', 'full')

%if ~exist('conv_data.mat')
    %[mini, full, lens] = test_img_size();
    %save('conv_data.mat', 'mini', 'full')
%else
    %load conv_data.mat
%end

loadExperimentParams;
filter_size = params.SCALE_PYRAMID;
lw = 2;

%plot(filter_size, mini.single.fft, 'b', 'DisplayName', 'single fft', 'LineWidth', lw)
%hold on;
%plot(filter_size, mini.single.fft_gpu, 'r', 'DisplayName', 'single fft gpu', 'LineWidth', lw)
%%plot(filter_size, mini.single.imf, 'DisplayName', 'single imf', 'LineWidth', lw)
%plot(filter_size, mini.single.imf_gpu, 'g', 'DisplayName', 'single imf gpu', 'LineWidth', lw)
%hold off
%legend('Location', 'northwest');
%tname = sprintf('Single precision %d X %d X %d', lens);
%title(tname)
%ylabel('Time (s)')
%xlabel('Filter size')
%%saveas(gcf, tname);

%%% SHOW RESULTS FOR SMALL IMAGE
%figure
%plot(filter_size, mini.single.fft, 'b', 'DisplayName', 'single fft', 'LineWidth', lw)
%hold on;
%plot(filter_size, mini.single.fft_gpu, 'r', 'DisplayName', 'single fft gpu', 'LineWidth', lw)
%%plot(filter_size, mini.single.imf, 'DisplayName', 'single imf', 'LineWidth', lw)
%plot(filter_size, mini.single.imf_gpu, 'g', 'DisplayName', 'single imf gpu', 'LineWidth', lw)
%%plot(filter_size, mini.double.fft, 'b', 'LineStyle', '--', 'DisplayName', 'double fft', 'LineWidth', lw)
%%plot(filter_size, mini.double.fft_gpu, 'r', 'LineStyle', '--', 'DisplayName', 'double fft gpu', 'LineWidth', lw)
%%%plot(filter_size, mini.double.imf, 'DisplayName', 'double imf', 'LineWidth', lw)
%%plot(filter_size, mini.double.imf_gpu, 'g', 'LineStyle', '--', 'DisplayName', 'double imf gpu', 'LineWidth', lw)
%hold off
%legend('Location', 'northwest');
%tname = sprintf('Convolve %d X %d X %d', lens);
%title(tname)
%ylabel('Time (s)')
%xlabel('Filter size')
%%saveas(gcf, tname);

%% SHOW RESULTS FOR FULL IMAGE
%figure
%plot(filter_size, full.single.fft, 'b', 'DisplayName', 'single fft', 'LineWidth', lw)
%hold on;
%plot(filter_size, full.single.fft_gpu, 'r', 'DisplayName', 'single fft gpu', 'LineWidth', lw)
%%plot(filter_size, full.single.imf, 'DisplayName', 'single imf', 'LineWidth', lw)
%plot(filter_size, full.single.imf_gpu, 'g', 'DisplayName', 'single imf gpu', 'LineWidth', lw)
%plot(filter_size, full.double.fft, 'b', 'LineStyle', '--', 'DisplayName', 'double fft', 'LineWidth', lw)
%plot(filter_size, full.double.fft_gpu, 'r', 'LineStyle', '--', 'DisplayName', 'double fft gpu', 'LineWidth', lw)
%%plot(filter_size, full.double.imf, 'DisplayName', 'double imf', 'LineWidth', lw)
%plot(filter_size, full.double.imf_gpu, 'g', 'LineStyle', '--', 'DisplayName', 'double imf gpu', 'LineWidth', lw)
%hold off
%legend('Location', 'southeast');
%tname = 'Convolve 2048 X 2048 X 141';
%title(tname)
%ylabel('Time (s)')
%xlabel('Filter size')
%saveas(gcf, tname);

%% SHOW DOUBLE PRECISION
%figure
%plot(filter_size, mini.double.fft, 'b', 'DisplayName', 'double fft', 'LineWidth', lw)
%hold on;
%plot(filter_size, mini.double.fft_gpu, 'r', 'DisplayName', 'double fft gpu', 'LineWidth', lw)
%%plot(filter_size, mini.double.imf, 'DisplayName', 'double imf', 'LineWidth', lw)
%plot(filter_size, mini.double.imf_gpu, 'g', 'DisplayName', 'double imf gpu', 'LineWidth', lw)
%hold off
%legend('Location', 'northwest');
%tname = sprintf('Double precision %d X %d X %d', lens);
%title(tname)
%ylabel('Time (s)')
%xlabel('Filter size')
%%saveas(gcf, tname);

function [mini, full, lens, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_img_size()

    mini = {}; full = {}; 
    fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
    img = load3DTif_uint16(fn);
    lens = [1024, 1024, 126]; % simulate a downsample; changeable for testing purposes
    img_mini = img(1:lens(1), 1:lens(2), 1:lens(3));

    mini = struct;

    [double_times, single_times, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_convn_dtype(img_mini);
    mini.double = double_times;
    mini.single = single_times;

    %full = struct;

    %[double_times, single_times] = test_convn_dtype(img);
    %full.double = double_times;
    %full.single = single_times;
end

function [double_times, single_times, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_convn_dtype(img)
    i_size = size(img);
    fprintf('\n\nTesting with size: %d, %d, %d\n', i_size)
    %double_times = test_convn_fsize(img);
    double_times = [];
    fprintf('\n\nConvert to SINGLE / FLOAT precision\n\n'); tic; img = single(img); toc;
    [single_times, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_convn_fsize(img);
end

function [times, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_convn_fsize(img)
    loadExperimentParams;
    filter_sizes = params.SCALE_PYRAMID;

    fft_times = [];
    fft_pad_times = [];
    fft_times_gpu = [];
    imf_times = [];
    imf_times_gpu = [];
    cuda_times = [];
    sep_times = [];

    for fsize = filter_sizes
        fprintf('\nFilter size: %d\n', fsize);
        h = fspecial3('gaussian', fsize);
        if strcmp(class(img), 'single') % match it
            h = single(h);
        end
        [t_fft, t_cuda, t_sep, t_fft_pad, t_fft_gpu, t_imf, t_imf_gpu, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_convn_vers(img, h);
        fft_times = [fft_times t_fft];
        cuda_times = [cuda_times t_cuda];
        fft_pad_times = [fft_pad_times t_fft_pad];
        fft_times_gpu = [fft_times_gpu t_fft_gpu];
        imf_times = [imf_times t_imf];
        imf_times_gpu = [imf_times_gpu t_imf_gpu];
        sep_times = [sep_times t_sep];
    end

    times = struct;
    times.fft = fft_times;
    times.fft_pad = fft_pad_times;
    times.imf = imf_times;
    times.fft_gpu = fft_times_gpu;
    times.imf_gpu = imf_times_gpu;
    times.cuda = cuda_times;
    times.sep = sep_times;
end

function [t_fft, t_cuda, t_sep, t_fft_pad, t_fft_gpu, t_imf, t_imf_gpu, img_blur_fft, img_blur_cuda, img_blur_cuda_1GPU] = test_convn_vers(img, h)

    assert(strcmp(class(img), class(h)))
    compute_err = @(X, ref) sum(sum(sum(abs(X - ref)))) / sum(ref(:));
    %fprintf('Type: %s\n', class(img_mini));
    t_fft = 0;
    t_fft_pad = 0;
    t_fft_gpu = 0;
    t_imf = 0;
    t_sep = 0;
    t_imf_gpu = 0;
    t_cuda = 0;
    img_blur_cuda_1GPU = [];

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

    %gpuDevice(1); % reset GPU avail mem
    %tic;
    %h1d = gpuArray(h(1, :));
    %img_gpu = gpuArray(img);
    %img_blur_sep = conv3_sep(img_gpu, h1d);
    %t_sep = toc;
    %%err = compute_err(img_blur_sep, img_blur_fft);
    %err = 0.0;
    %fprintf('`conv3_sep` %s: %.4f rel. error %.2f\n', class(img_gpu), t_sep, err)

    %gpuDevice(1); % reset GPU avail mem
    %tic;
    %h1d = h(1, :);
    %img_blur_seplib = convnsep({h1d, h1d, h1d}, img, 'same', 1500);
    %t_seplib = toc;
    %err = compute_err(img_blur_seplib, img_blur_fft);
    %fprintf('`convsep` %s: %.4f rel. error %.2f\n', class(img_gpu), t_seplib, err)

    % Multi GPU section
    gpuDevice(1); % reset GPU avail mem
    tic;
    img_blur_cuda = convn_cuda(img, h, 1);
    t_cuda = toc;
    err = compute_err(img_blur_cuda, img_blur_fft);
    norm_err = norm(img_blur_fft(:)-img_blur_cuda(:)) / norm(img_blur_fft(:))
    fprintf('`convn_cuda` %s: %.4f r err %e n err. %e\n', class(img), t_cuda, err, norm_err)
    assert(isequal(size(img_blur_fft), size(img_blur_cuda)));
    gpuDevice();

    %% 1 GPU section; DEPRECATED
    %gpuDevice(1); % reset GPU avail mem
    %tic;
    %img_blur_cuda_1GPU = convn_1GPU_cuda(img, h);
    %t_cuda_1GPU = toc;
    %err = compute_err(img_blur_cuda_1GPU, img_blur_fft);
    %norm_err = norm(img_blur_fft(:)-img_blur_cuda_1GPU(:)) / norm(img_blur_fft(:))
    %fprintf('`convn_1GPU_cuda` %s: %.4f r err %e n err. %e\n', class(img), t_cuda_1GPU, err, norm_err)
    %assert(isequal(size(img_blur_fft), size(img_blur_cuda_1GPU)));
    %norm(img_blur_fft(:)-img_blur_cuda_1GPU(:))/ norm(img_blur_fft(:))
    %gpuDevice();

    %options.Power2Flag = true;
    %tic; 
    %img_blur_fft_pad = convnfft(img, h, 'same', [], options); 
    %t_fft_pad = toc;
    %fprintf('`convnfft` power2flag true %s: %.4f\n', class(img), t_fft_pad)

    %options.GPU = true;
    %gpuDevice(1); % reset GPU avail mem
    %%profile on -history; 
    %try
        %tic; 
        %img_blur_fft = convnfft(img, h, 'same', [], options); 
        %t_fft_gpu = toc;
        %fprintf('`convnfft` gpuArray power2flag false %s: %.4f\n', class(img), t_fft_gpu)
    %catch
        %void = toc;
        %t_fft_gpu = 0;
        %disp('GPU `convnfft` failed, out of memory')
    %end

    %profile off; 
    %profsave(profile('info'), sprintf('pr-convnfft-gpu-%s-fsize%d', class(img), fsize(1))); 

    %gpuDevice(1); % reset GPU avail mem
    %tic; 
    %try
        %img_filter_func = convn_filter(img, h); 
        %t_imf_gpu = toc;
        %fprintf('`imfilter` gpuArray %s: %.4f\n', class(img), t_imf_gpu);
    %catch
        %void = toc;
        %disp('GPU `convn_filter` failed, out of memory')
        %%disp('GPU `convn_filter` failed, out of memory. Timing CPU:')
        %%tic;
        %%try
            %%timeout('imfilter(img, h, ''same'', ''conv'')', timeout_len)
            %%t_imf = toc;
        %%catch
            %%void = toc;
            %%fprintf('`imfilter` exceeded timeout length: %d', timeout_len)
            %%t_imf = 0;
        %%end
        %%fprintf('`imfilter` cpu %s: %.4f\n', class(img), t_imf);
    %end
    
    %gpuDevice(1);
    %tic;
    %imfilter(gpuArray(img), gpuArray(h), 'same', 'conv');
    %t_imf = toc;
    %fprintf('`imfilter` gpu %s: %.4f\n', class(img), t_imf);

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
