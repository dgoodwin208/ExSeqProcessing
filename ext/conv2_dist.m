
% distribute the convolution function, chunking to avoid GPU memory errors
function J = conv2_dist(filter, J, chunk_cols)
% FIXME gpu_chunks chosen arbitrarily to split convolution up

[row, cols] = size(J);
%chunk_cols = ceil(cols / gpu_chunks);
gpu_chunks = ceil(cols / chunk_cols); % adjust actual chunks

filter = gpuArray(filter);

% precompute indices to remove if/else from loop
indices = zeros(gpu_chunks, 2);
start = 1;
for i=1:gpu_chunks
    indices(i, :) = [start, start + chunk_cols - 1];
    start = start + chunk_cols;
end
indices(end, 2) = cols; % never out of bound

%cols
%indices(end-1,:)
%indices(end,:)

for i=1:gpu_chunks
    %s = gpuDevice(); sprintf('GPU mem: %.2f\n', s.AvailableMemory / 1e9)
    %sprintf('chunkcols:%d, totalchunks: %d, chunknum: %d, total cols: %d, start: %d, stop: %d\n', chunk_cols, gpu_chunks, i, cols, indices(i, 1), indices(i, 2))
    J_tile = gpuArray(J(:, indices(i,1):indices(i,2)));
    J(:, indices(i,1):indices(i,2)) = gather(conv2(filter, 1, J_tile, 'same'));
    clear J_tile;
end

clear filter
end

