function mask = meshtovoxels(varargin)
    % meshtovoxels Voxelize a closed mesh
    %
    %   Syntax
    %   ------
    %   mask = meshtovoxels('Faces', F, 'Vertices', V, 'XGridVector', X, 'YGridVector', Y, 'ZGridVector', Z)
    %   mask = meshtovoxels('f', F, 'v', V, 'x', X, 'y', Y, 'z', Z)
    %
    %   Description
    %   -----------
    %   mask = meshtovoxels('f', F, 'v', V, 'x', X, 'y', Y, 'z', Z)
    %   generates a Boolean mask from the mesh represented by
    %   faces/triangles F and vertices V using the grid vectors X, Y and Z.
    %   The faces input F is an mx3 or 3xm array that specifies the
    %   vertices to connect in the vertices array V. X, Y and Z represent
    %   the grid vectors to use to create the meshgrid for voxel creation.
    %
    %   Notes
    %   -----
    %   meshtovoxels tests for overlap of voxels' and triangles'
    %   axis-aligned bounding boxes (AABB). Then, triangles' edges are
    %   tested for overlap with voxels that pass the AABB test. Voxels that
    %   pass both tests are filled. meshtovoxels voxelises a shell
    %   corresponding to the mesh; it does not fill the inside of the mesh.
    %
    %   For more information, see <a href="matlab: 
    %   web('https://developer.nvidia.com/content/basics-gpu-voxelization')
    %   ">The Basics of GPU Voxelization</a>
    %   and <a href="matlab:
    %   web('https://dl.acm.org/citation.cfm?doid=1882261.1866201')
    %   ">Fast parallel surface and solid voxelization on GPUs</a>
    %
    %   © 2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also isosurface | patch
    
    %% Parse the inputs.
    parser = inputParser;
    
    parser.addParameter('faces', [], ...
        @(f)ismatrix(f) && any(size(f) == 3))
    parser.addParameter('vertices', [], ...
        @(v)ismatrix(v) && any(size(v) == 3))
    parser.addParameter('xgridvector', [], @(X)isvector(X))
    parser.addParameter('ygridvector', [], @(Y)isvector(Y))
    parser.addParameter('zgridvector', [], @(Z)isvector(Z))
    
    parser.parse(varargin{:})
    
    %% Create the triangles array.
    faces = parser.Results.faces;
    vertices = parser.Results.vertices;
    
    % Convert the arrays to mx3 if necessary.
    if size(faces, 1) == 3
        faces = transpose(faces);
    end % if
    
    if size(vertices, 1) == 3
        vertices = transpose(vertices);
    end % if
    
    triangles = cat(3, ...
        vertices(faces(:, 1), :), ...
        vertices(faces(:, 2), :), ...
        vertices(faces(:, 3), :));
    
    % Calculate the triangle extents.
    triMin = min(triangles, [], 3);
    triMax = max(triangles, [], 3);
    
    %% Create the grids.
    xVector = parser.Results.xgridvector;
    yVector = parser.Results.ygridvector;
    zVector = parser.Results.zgridvector;
    
    dp = [...
        xVector(2) - xVector(1), ...
        yVector(2) - yVector(1), ...
        zVector(2) - zVector(1)];
    
    [xGrid, yGrid, zGrid] = meshgrid(xVector, yVector, zVector);
    
    %% Calculate the mesh bounding box mask.
    meshMin = min(vertices, [], 1);
    meshMax = max(vertices, [], 1);

    box = ...
        xGrid >= meshMin(1) - dp(1) & xGrid <= meshMax(1) + dp(1) & ...
        yGrid >= meshMin(2) - dp(2) & yGrid <= meshMax(2) + dp(2) & ...
        zGrid >= meshMin(3) - dp(3) & zGrid <= meshMax(3) + dp(3);
    
    % Only test voxels in the bounding box.
    idxBox = find(box);
    pV = [xGrid(idxBox), yGrid(idxBox), zGrid(idxBox)];
    
    %% Allocate the mask.
    mask = false(size(box));
    
    %% For every triangle, test voxels for AABB overlap, then edge overlap.
    for t = 1:size(triangles, 1)
        %% Find voxels that overlap the triangle's 2D projections.
        xyOverlap = ...
            (triMax(t, 1) - pV(:, 1)).*(triMin(t, 1) - pV(:, 1) - dp(1)) < 0 & ...
            (triMax(t, 2) - pV(:, 2)).*(triMin(t, 2) - pV(:, 2) - dp(2)) < 0;
        
        yzOverlap = ...
            (triMax(t, 2) - pV(:, 2)).*(triMin(t, 2) - pV(:, 2) - dp(2)) < 0 & ...
            (triMax(t, 3) - pV(:, 3)).*(triMin(t, 3) - pV(:, 3) - dp(3)) < 0;
        
        xzOverlap = ...
            (triMax(t, 1) - pV(:, 1)).*(triMin(t, 1) - pV(:, 1) - dp(1)) < 0 & ...
            (triMax(t, 3) - pV(:, 3)).*(triMin(t, 3) - pV(:, 3) - dp(3)) < 0;
        
        idxOverlap = idxBox(xyOverlap & yzOverlap & xzOverlap);
        
        if isempty(idxOverlap)
            continue
        end % if
        
        %% Construct the triangle edges and calculate the normal.
        v1 = triangles(t, :, 1);
        v2 = triangles(t, :, 2);
        v3 = triangles(t, :, 3);
        
        % Calculate the triangles edge vectors.
        e1 = v2 - v1;
        e2 = v3 - v2;
        e3 = v1 - v3;
        
        % Calculate the triangle plane normal.
        n = cross(e1, e2);
        sN = sign(n);
        sN(sN == 0) = 1;

        %% Find voxels that overlap the triangle plane.
        % Get the voxel coordinates.
        pVO = [xGrid(idxOverlap), yGrid(idxOverlap), zGrid(idxOverlap)];
        
        % Get the test points.
        c = dp;
        c(n <= 0) = 0;
        
        % Calculate the offsets.
        d1 = sum(n.*(c - v1));
        d2 = sum(n.*((dp - c) - v1));
        
        dnp = sum(repmat(n, [size(pVO, 1), 1]).*pVO, 2);
        
        isPlane = (dnp + d1).*(dnp + d2) <= 0;
        
        idxPlane = idxOverlap(isPlane);
        
        if isempty(idxPlane)
            continue
        end % if
        
        %% Setup edge-voxel intersection test.
        % XY plane:
        xyN = sN(3)*[...
            -e1(2), e1(1);
            -e2(2), e2(1);
            -e3(2), e3(1)];
        
        xyD = -sum(xyN.*[v1(1:2); v2(1:2); v3(1:2)], 2) + ...
            max(cat(2, [0; 0; 0], dp(1)*xyN(:, 1)), [], 2) + ...
            max(cat(2, [0; 0; 0], dp(2)*xyN(:, 2)), [], 2);
        
        % YZ plane:
        yzN = sN(1)*[...
            -e1(3), e1(2);
            -e2(3), e2(2);
            -e3(3), e3(2)];

        yzD = -sum(yzN.*[v1(2:3); v2(2:3); v3(2:3)], 2) + ...
            max(cat(2, [0; 0; 0], dp(2)*yzN(:, 1)), [], 2) + ...
            max(cat(2, [0; 0; 0], dp(3)*yzN(:, 2)), [], 2);

        %xz plane:
        xzN = -sN(2)*[...
            -e1(3), e1(1);
            -e2(3), e2(1);
            -e3(3), e3(1)];

        xzD = -sum(xzN.*[v1([1, 3]); v2([1, 3]); v3([1, 3])], 2) + ...
            max(cat(2, [0; 0; 0], dp(1)*xzN(:, 1)), [], 2) + ...
            max(cat(2, [0; 0; 0], dp(3)*xzN(:, 2)), [], 2);

        %% Check for overlap of the 2D projections.
        for v = idxPlane'
            xyP = repmat([xGrid(v), yGrid(v)], [3, 1]);
            yzP = repmat([yGrid(v), zGrid(v)], [3, 1]);
            xzP = repmat([xGrid(v), zGrid(v)], [3, 1]);

            if all(sum(xyN.*xyP, 2) + xyD >= 0) && ...
               all(sum(yzN.*yzP, 2) + yzD >= 0) && ...
               all(sum(xzN.*xzP, 2) + xzD >= 0)
                
                % Cover the voxel and remove it from the list of voxels to
                % test.
                mask(v) = true;
                
                pV(idxBox == v, :) = [];
                idxBox(idxBox == v) = [];
            end % if
        end % for v
    end % for t
end % meshtovoxels
