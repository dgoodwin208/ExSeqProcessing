
sem_name = sprintf('/%s.gc',getenv('USER'));
semaphore(sem_name,'open',1);
ret = semaphore(sem_name,'getvalue');
if ret ~= 1
    semaphore(sem_name,'unlink');
    semaphore(sem_name,'open',1);
end

%fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
%fn = fullfile('/mp/nas1/share/ExSEQ/AutoSeq2/xy01/3_normalization/exseqauto-xy01_round001_ch01SHIFT.tif');
fn = fullfile('3_normalization/exseqauto-xy01-downsample_round001_ch01SHIFT.tif');
img = load3DTif_uint16(fn);

LoadParams;
% saved keypoints for 1024 1024 126 image
load res_vect
keys = res_vect;

img = img(1:30, 1:90, 1:30);
keys = [15,45,15];

skipDescriptors = false;

tic
sift_keys_cuda = calculate_3DSIFT_cuda(img, keys, skipDescriptors);
cuda_time = toc;
N = length(sift_keys_cuda);
fprintf('Finished CUDA SIFT len %d in %.1f s\n', N, cuda_time);

tic
sift_keys = calculate_3DSIFT(img, keys, skipDescriptors);
M = length(sift_keys);
orig_time = toc;
fprintf('Finished SIFT len %d\n', length(sift_keys));

tic
total_error = 0;
total_energy = 0;
mismatch_remove = 0;
match = 0;
mismatch = 0;
for i=1:N
    cuda_key = sift_keys_cuda{i};

    % find matching key
    found = false;
    for j=1:M
        key = sift_keys{j};
        if (~isempty(key) && ~isnumeric(key) && (key.x == cuda_key.x) && (key.y == cuda_key.y) && (key.z == cuda_key.z))
            found = true;
            break;
        end
    end

    if found
        if skipDescriptors || isequal(cuda_key.ivec, key.ivec);
            %fprintf('Keypoint %d succeeded match\n', i);
            match = match + 1;
        else
            %fprintf('Keypoint %d failed match\n', i);
            mismatch = mismatch + 1;
            % tally energy of only fails
            for k=1:sift_params.descriptor_len
                total_energy = total_energy + double(abs(cuda_key.ivec(k)));
                if ~isequal(cuda_key.ivec(k), key.ivec(k))
                    %fprintf('sift_keys_cuda{%d}.ivec(%d)=%d, sift_keys{%d}.ivec(%d)=%d\n', ...
                        %i, k, cuda_key.ivec(k), j, k, key.ivec(k));
                    total_error = total_error + double(abs(cuda_key.ivec(k) - key.ivec(k)));
                end
            end
        end
    else
        fprintf('Keypoint %d removed in MATLAB version\n', i);
        mismatch_remove = mismatch_remove + 1;
    end
end
comparison = toc;

rel_error = 100 * double(total_error) / double(total_energy);
pmatch = 100 * match / N;
fprintf('Run N=%d TwoPeak %d fv real\n\tPercent match: %.5f\n\tPercent error among mismatches: %.5f\n\tmatch: %d, mismatch: %d, mismatched removes: %d \n', ...
    N, sift_params.TwoPeak_Flag, pmatch, rel_error, match, mismatch, mismatch_remove);
fprintf('\tCuda time: %.1f Original time: %.1f Comparison: %.1f\n', cuda_time, orig_time, comparison);

semaphore(sem_name,'unlink');
