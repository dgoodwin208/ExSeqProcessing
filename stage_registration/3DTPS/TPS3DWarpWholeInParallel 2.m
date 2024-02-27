%TPS3DWARP Calculates a 3D Thin Plate Spline, then applies it to the moving
%Tile input (tileM). Keypoints for moving (keyM) and fixed (keyF) are
%intended to outputs from the calc_affine function, ie, vetted
%correspondence points that are stored as a Nx3 vector format. outDim is
%the size of the (padded) output image.

%This function returns the indices of the input image (in1D_total) that map to
%the output image indices

function [ in1D_total, out1D_total ] = TPS3DWarpWholeInParallel(keyM,keyF, inDim, outDim)

loadParameters;

%How many slices to make of tileM when calculating the warp? (this is a
%heuristic based on figuring out fastest speed. Doing it without the breaks
%takes indefinitely long)
TARGET_CHUNK_SIZE = 150;
ZRES = 5; %%split the z direction manually

y_grid = splitArrayIndices(inDim,1, TARGET_CHUNK_SIZE); 
x_grid = splitArrayIndices(inDim,2, TARGET_CHUNK_SIZE);
z_grid = splitArrayIndices(inDim,3, ZRES);

y_grid(end) = y_grid(end) + 1;
x_grid(end) = x_grid(end) + 1;
z_grid(end) = z_grid(end) + 1;


%Preallocate the resulting output to avoid memory issues
%To do so just get a sense of big it will be:
total_iterations = (length(y_grid)-1)*(length(x_grid)-1)*(length(z_grid)-1);





tic;
%======================================================
% Calculate Parameters for Thin Plate Spline
% Using code from Yang Yang available at this link:
% http://www.mathworks.com/matlabcentral/fileexchange/47409-glmdtps-registration-method/content/GLMD_Demo/src/TPS3D.m
% The fundamental calculation code is his, but his code was restructured to fit the SWITCH context
% Specifically, the chunking we had to do to handle how big the SWITCH datasets are
%======================================================
%if isfield(params,'TPS3DWARP_MAX_THREADS')
%    fprintf('set threads\n');
%    maxNumCompThreads(params.TPS3DWARP_MAX_THREADS);
%end
npnts = size(keyM,1);
K = zeros(npnts, npnts);
for rr = 1:npnts
    for cc = 1:npnts
        K(rr,cc) = sum( (keyM(rr,:) - keyM(cc,:)).^2 ); %R^2 
        K(cc,rr) = K(rr,cc);
    end;
end;
%calculate kernel function R
K = max(K,1e-320); 
%K = K.* log(sqrt(K));
K = sqrt(K); %
% Calculate P matrix
P = [ones(npnts,1), keyM]; %nX4 for 3D
% Calculate L matrix
L = [ [K, P];[P', zeros(4,4)] ]; %zeros(4,4) for 3D
K = [];
P = [];
param = pinv(L) * [keyF; zeros(4,3)]; %zeros(4,3) for 3D
L = [];
%======================================================
% done
%======================================================
toc;
%

grid_idx = zeros(total_iterations, 3);
i = 1;
for y = 1:length(y_grid)-1
    for x = 1:length(x_grid)-1
        for z = 1:length(z_grid)-1
            grid_idx(i, :) = [y, x, z];
            i = i+1;
        end
    end
end

%Keep track of all X->X' indices, after the full map is created, the input
%images can be mapped into the 

out1D = cell(1, total_iterations);
in1D = cell(1, total_iterations);

disp('start parfor loop in TPS3DWarpWhole')
tic;

if isfield(params,'TPS3DWARP_MAX_THREADS')
    worker_max_threads = params.TPS3DWARP_MAX_THREADS;
else
    worker_max_threads = 'automatic';
end
parfor idx = 1:total_iterations
    maxNumCompThreads(worker_max_threads);

    y = grid_idx(idx, 1);
    x = grid_idx(idx, 2);
    z = grid_idx(idx, 3);

    %note the switch of X and Y designations here. 
    [X, Y, Z] = meshgrid( y_grid(y):(y_grid(y+1)-1), x_grid(x):(x_grid(x+1)-1), z_grid(z):(z_grid(z+1)-1) );
    inputgrid = [X(:)'; Y(:)'; Z(:)']';
 


    %======================================================
    % Implement TP Spline
    % Using code from Yang Yang available at this link:
    % http://www.mathworks.com/matlabcentral/fileexchange/47409-glmdtps-registration-method/content/GLMD_Demo/src/TPS3D.m
    % The fundamental calculation code is his, but his code was restructured to fit the SWITCH context
    % Specifically, the chunking we had to do to handle how big the SWITCH datasets are
    %======================================================
    pntsNum=size(inputgrid,1); 

    K = pdist2(inputgrid, keyM, 'euclidean'); %|R| for 3D
    P = [ones(pntsNum,1), inputgrid(:,1), inputgrid(:,2), inputgrid(:,3)]; % quaternion of inputgrid
    outputgrid = K * param(1:npnts,:) + P * param((end-3):end,:);
    K = [];
    P = [];

    outputgrid(:,1)=round(outputgrid(:,1)*10^3)*10^-3;
    outputgrid(:,2)=round(outputgrid(:,2)*10^3)*10^-3;
    outputgrid(:,3)=round(outputgrid(:,3)*10^3)*10^-3;
    %======================================================
    % done
    %======================================================
    
    %copy output directly into output image
    %Note: 1D is SIGNFICANTLY faster (orders of magnitude) than Nd
    o = round(outputgrid); i = round(inputgrid);
    problem_indices = [];
    try
    problem_indices = unique([find(o(:,1)>outDim(1));find(o(:,2)>outDim(2)); find(o(:,3)>outDim(3)); ...
        find(o(:,1)<1);find(o(:,2)<1); find(o(:,3)<1)]);
    catch
        disp('aah');
    end
    if ~isempty(problem_indices)
        
        %disp(['ERROR: out of bounds. Chopping out ' num2str(length(problem_indices)/size(o,1)) '% of the pixels ']);
        
        o(problem_indices,:) = []; %this is short for removing elemnts
        i(problem_indices,:) = [];
        
    end
    if size(o,2)<3
        disp('aaah');
    end
    
    %keeping everything in 1D for speed, this method works but then
    %needs later interpolation
    out1D{idx} = sub2ind(outDim,o(:,1),o(:,2),o(:,3));
    try
    in1D{idx} = sub2ind(inDim,i(:,1),i(:,2),i(:,3));            
    catch
       disp('ah'); 
    end
    
    if mod(idx,100)==0 
    disp([num2str(idx) '/' num2str(total_iterations) ' Completed one round'])
    end
end

disp('finished parfor loop in TPS3DWarpWhole')
toc;

out1D_total = zeros(prod(outDim),1); out1d_ptr = 1;
in1D_total = zeros(prod(inDim),1); in1d_ptr = 1;

% gather each result from workers
for idx = 1:total_iterations
    if ~isempty(out1D{idx})
        out1D_total(out1d_ptr:out1d_ptr+length(out1D{idx})-1) = out1D{idx};
        in1D_total(out1d_ptr:in1d_ptr+length(in1D{idx})-1) = in1D{idx};

        out1d_ptr = out1d_ptr+length(out1D{idx});
        in1d_ptr = in1d_ptr+length(in1D{idx});
    end
end

%truncate the in1d and out1d only to the used results
out1D_total = out1D_total(1:out1d_ptr-1);
in1D_total = in1D_total(1:in1d_ptr-1);

end



