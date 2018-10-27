
% distribute the convolution function, chunking to avoid GPU memory errors
function J = conv2_dist(filter, J, chunk_cols, gpu_strategy)
% FIXME gpu_chunks chosen arbitrarily to split convolution up

if ~gpu_strategy
    gpu_strategy = 'single'
elseif ~strcmp(gpu_strategy, 'multi') && ~strcmp(gpu_strategy, 'single')
    error('gpu_strategy must be either single of multi')
end

[row, cols] = size(J);
gpu_chunks = ceil(cols / chunk_cols); % adjust actual chunks

%FIXME? the host should not have access?
filter = gpuArray(filter);

% precompute indices to remove if/else from loop
indices = zeros(gpu_chunks, 2);
start = 1;
for i=1:gpu_chunks
    indices(i, :) = [start, start + chunk_cols - 1];
    start = start + chunk_cols;
end
indices(end, 2) = cols; % never out of bound

if strcmp(gpu_strategy, 'single')
    for i=1:gpu_chunks
        %s = gpuDevice(); sprintf('GPU mem: %.2f\n', s.AvailableMemory / 1e9)
        %sprintf('chunkcols:%d, totalchunks: %d, chunknum: %d, total cols: %d, start: %d, stop: %d\n', ...
            % chunk_cols, gpu_chunks, i, cols, indices(i, 1), indices(i, 2))
        J_tile = gpuArray(J(:, indices(i,1):indices(i,2)));
        J(:, indices(i,1):indices(i,2)) = gather(conv2(filter, 1, J_tile, 'same'));
        clear J_tile;
    end
elseif strcmp(gpu_strategy, 'multi')

    %% start a parallel pool for each GPU
    %tic
    %gpuDevice([]); % make all memory on devices completely available
    %nGPUs = gpuDeviceCount();
    nGPUs = 3;
    %parpool('local', nGPUs);
    %toc

    % set which gpu to process which chunks
    chunks_per_gpu = ceil(gpu_chunks / nGPUs);
    gpu_indices = zeros(nGPUs, 2);
    start = 1;
    for i=1:nGPUs
        gpu_indices(i, :) = [start, start + chunks_per_gpu - 1];
        start = start + chunks_per_gpu;
    end
    gpu_indices(end, 2) = cols; % never out of bound
    gpu_indices

    %J_array = cell(gpu_chunks);
    %for i = 1:gpu_chunks
        %J_array(i) = J(:, indices(i,1):indices(i,2));
    %end

    % parpool
    for i = 1:nGPUs
        %J_array(i) = gpu_conv2(J_array(i), filter);
        for j=gpu_indices(i, 1):gpu_indices(i, 2)
            J(:, indices(j,1):indices(j,2)) = gpu_conv2(J(:, indices(j,1):indices(j,2)), filter);
            %gpu_conv2(J(:, indices(j,1):indices(j,2)), filter);
            %J_array{i} = gpu_conv2(J(:, indices(j,1):indices(j,2)), filter);
        end
    end
end

clear filter
end

function out = gpu_conv2(J_tile, filter)
    J_tile = gpuArray(J_tile);
    out = gather(conv2(filter, 1, J_tile, 'same'));
    clear J_tile;
end

