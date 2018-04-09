
fn = fullfile('/mp/nas1/share/ExSEQ/ExSeqAutoFrameA1/3_normalization/exseqautoframea1_round006_ch03SHIFT.tif');
len = 400;
img = load3DTif_uint16(fn);

N = 10000
%keypts = zeros(N, 3);
load res_vect

res_vect_rest = res_vect(1:N, :);

tic
new_keys = calculate_3DSIFT(img, res_vect_rest, false);
toc

%load 3DSIFTkeys % loads old keys

%assert(isequal(new_keys, keys))

%start= 1;
%for i=1:N
    %keypts(i, 1) = start;
    %start = start + 120;
%end
%keypts

%keys = calculate_3DSIFT(img, keypts, false);
