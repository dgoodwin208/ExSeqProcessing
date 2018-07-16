
%fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
fn = fullfile('/mp/nas1/share/ExSEQ/AutoSeq2/xy01/3_normalization/exseqauto-xy01_round001_ch01SHIFT.tif');
img = load3DTif_uint16(fn);

N = 1;
%keypts = zeros(N, 3);
% saved keypoints for 2048 2048 141 image
load res_vect

keys = res_vect(1:N, :);
%keys = res_vect;
%keys(1)
%keys(2)

%sub2ind(size(img), keys(1,1), keys(1,2), keys(1,3));
%sub2ind(size(img), keys(2,1), keys(2,2), keys(2,3));

tic
sift_keys_cuda = calculate_3DSIFT_cuda(img, keys, true);
toc
fprintf('Finished cuda sift\n');

%tic
%sift_keys = calculate_3DSIFT(img, keys, false);
%toc

%load 3DSIFTkeys % loads old keys

%assert(isequal(new_keys, keys))

%start= 1;
%for i=1:N
    %keypts(i, 1) = start;
    %start = start + 120;
%end
%keypts

