
%fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
fn = fullfile('/mp/nas1/share/ExSEQ/AutoSeq2/xy01/3_normalization/exseqauto-xy01_round001_ch01SHIFT.tif');
img = load3DTif_uint16(fn);
%img = 1:27;
%img = reshape(img, 3,3,3);
%keys = [2,2,2];

LoadParams;
% saved keypoints for 2048 2048 141 image
load res_vect

% keypoint 9 is rejected
keys = res_vect(1:1000, :)
%keys = res_vect(9, :)

skipDescriptors = false;

tic
sift_keys_cuda = calculate_3DSIFT_cuda(img, keys, skipDescriptors);
toc
fprintf('Finished CUDA SIFT\n');
N = length(sift_keys_cuda);

tic
sift_keys = calculate_3DSIFT(img, keys, skipDescriptors);
toc
fprintf('Finished SIFT\n');

rel_error = 0;
total_energy = 0;
mismatch_remove = 0;
for i=1:N
    cuda_key = sift_keys_cuda{i};

    % find matching key
    found = false;
    for j=1:N
        key = sift_keys{j};
        if (~isempty(key) && ~isnumeric(key) && (key.x == cuda_key.x) && (key.y == cuda_key.y) && (key.z == cuda_key.z))
            found = true;
            break;
        end
    end

    if found
        for k=1:sift_params.descriptor_len
            total_energy = total_energy + abs(cuda_key.ivec(k));
            if ~isequal(cuda_key.ivec(k), key.ivec(k))
                fprintf('sift_keys_cuda{%d}.ivec(%d)=%d, sift_keys{%d}.ivec(%d)=%d\n', ...
                i, k, cuda_key.ivec(k), j, k, key.ivec(k));
                rel_error = rel_error + abs(cuda_key.ivec(k) - key.ivec(k));
            end
        end
        assert(isequal(cuda_key.ivec', key.ivec));
        fprintf('Keypoint %d succeeded\n', i);
    else
        fprintf('Keypoint %d removed in MATLAB version\n', i);
        mismatch_remove = mismatch_remove + 1;
    end
end

rel_error = rel_error / total_energy;
fprintf('Relative error: %.5f, mismatched removes: %d\n', rel_error, mismatch_remove);

%load 3DSIFTkeys % loads old keys

%assert(isequal(new_keys, keys))

%start= 1;
%for i=1:N
    %keypts(i, 1) = start;
    %start = start + 120;
%end
%keypts

