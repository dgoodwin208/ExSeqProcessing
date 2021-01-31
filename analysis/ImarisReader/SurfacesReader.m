classdef SurfacesReader < SurpassObjectReader
    % SurfacesReader Read Imaris ims file Surfaces data
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader | SurpassObjectReader

    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        NumberOfSurfaces % Number of surfaces in the object
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    methods
        
        function obj = SurfacesReader(GID)
            % SurfacesReader Create a Surfaces reader object
            %
            %   obj = SurfacesReader(GID) constructs obj to read Imaris
            %   Surfaces data. GID is an HDF5 Group Identifier for a
            %   /Surfaces group in an ims file.
            
            %% Call the superclass constructor.
            obj = obj@SurpassObjectReader(GID, 'Surfaces');
            
            %% Check for the required data sets for a valid Spots object.
            isValidSurfaces = ...
                H5L.exists(GID, 'TimeInfos', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'TimeNVerticesNTriangles', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'Triangles', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'Vertices', 'H5P_DEFAULT');
            
            %% If the group is a valid Surfaces group, determine the number of surfaces.
            if isValidSurfaces
                DID = H5D.open(GID, 'TimeNVerticesNTriangles');
                FSID = H5D.get_space(DID);
                [~, spaceDims, ~] = H5S.get_simple_extent_dims(FSID);
                obj.NumberOfSurfaces = spaceDims(1);
        
                % Close the HDF5 dataset objects.
                H5S.close(FSID);
                H5D.close(DID)
            end % if
        end % SurfacesReader
        
        function ids = GetIDs(obj)
            % GetIDs Get the IDs for all surfaces
            %
            %   ids = obj.GetIDs returns a vector containing the IDs of all
            %   the surfaces in the Surfaces object.
            
            %% Read the ID field from the Surface dataset.
            DID = H5D.open(obj.GIDS8, 'Surface');
            FSID = H5D.get_space(DID);
            TID = H5T.create('H5T_COMPOUND', 8);
            H5T.insert(TID, 'ID', 0, 'H5T_STD_I64LE');
            
            dataID = H5D.read(DID, TID, 'H5S_ALL', FSID, 'H5P_DEFAULT');
            
            %% Close the HDF5 objects.
            H5T.close(TID)
            H5S.close(FSID)
            H5D.close(DID)
            
            %% Return the IDs.
            ids = dataID.ID;
        end % GetIDs
        
        function idx = GetIndicesT(obj)
            % GetIndicesT Get the time indices of the surfaces
            %
            %   idx = obj.GetIndicesT returns a vector of zero-based
            %   indices indicating the time index of each surface.

            %% Open the Time dataset.
            DID = H5D.open(obj.GID, 'TimeNVerticesNTriangles');
            dataTimeNVerticesNTriangles = transpose(H5D.read(DID));
            idx = dataTimeNVerticesNTriangles(:, 1);
            
            %% Close the HDF5 dataset object.
            H5D.close(DID)
        end % GetIndicesT
        
        function mask = GetMask(obj, sIdx)
            % GetMask Get the voxel mask for a surface object
            %
            %   mask = obj.GetMask(sIdx) returns a 3D logical array
            %   representing the surface as a voxel mask. The input sIdx
            %   indicates the zero-based index of the surface in the
            %   Surfaces object.
            
            %% Get the Imaris dataset grid. 
            FID = H5I.get_file_id(obj.GID);
            infoGID = H5G.open(FID, '/DataSetInfo');
            imageGID = H5G.open(infoGID, 'Image');
            
            % Read the xyz dimensions.
            aID = H5A.open(imageGID, 'X');
            xSize = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'Y');
            ySize = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'Z');
            zSize = str2double(H5A.read(aID));
            H5A.close(aID)
            
            % Read the extents.
            aID = H5A.open(imageGID, 'ExtMin0');
            xMin = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMax0');
            xMax = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMin1');
            yMin = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMax1');
            yMax = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMin2');
            zMin = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMax2');
            zMax = str2double(H5A.read(aID));
            H5A.close(aID)
            
            % Close the HDF5 objects.
            H5G.close(imageGID);
            H5G.close(infoGID);
            H5F.close(FID)
            
            %% Construct the grid vectors.
            xgridvector = linspace(xMin, xMax, xSize + 1);
            ygridvector = linspace(yMin, yMax, ySize + 1);
            zgridvector = linspace(zMin, zMax, zSize + 1);
            
            %% Call the meshtovoxels function to create the mask.
            mask = meshtovoxels(...
                'f', obj.GetTriangles(sIdx) + 1, ...
                'v', obj.GetVertices(sIdx), ...
                'x', xgridvector(1:end - 1), ...
                'y', ygridvector(1:end - 1), ...
                'z', zgridvector(1:end - 1));
        end % GetMask
        
        function normals = GetNormals(obj, sIdx)
            % GetNormals Get the normals for a surface object's vertices
            %
            %   normals = obj.GetNormals(sIdx) returns an mx3 array of
            %   normals for the surface represented by the zero-based index
            %   sIdx.
            
            %% If the object doesn't have calculated statistics, return.
            if isempty(obj.GIDS8)
                normals = [];
                return
            end % if
            
            %% Get the number of triangles and vertices dataset.
            DID = H5D.open(obj.GID, 'TimeNVerticesNTriangles');
            dataTimeNVerticesNTriangles = transpose(H5D.read(DID));
            H5D.close(DID)
            
            %% Calculate the offset and the slab to read for the surface.
            offset = sum(dataTimeNVerticesNTriangles(1:sIdx, 2), 1);
            slab = double(dataTimeNVerticesNTriangles(sIdx + 1, 2));
            
            %% Read the Vertex dataset slab for the surface.
            DID = H5D.open(obj.GIDS8, 'Vertex');

            FSID = H5D.get_space(DID);
            MSID = H5S.create_simple(1, slab, []);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);            
            
            TID = H5T.create('H5T_COMPOUND', 24);
            H5T.insert(TID, 'NormalX', 0, 'H5T_NATIVE_DOUBLE')
            H5T.insert(TID, 'NormalY', 8, 'H5T_NATIVE_DOUBLE')
            H5T.insert(TID, 'NormalZ', 16, 'H5T_NATIVE_DOUBLE')
            
            dataVertex = H5D.read(DID, TID, MSID, FSID, 'H5P_DEFAULT');
            
            %% Close the HDF5 objects.
            H5S.close(FSID)
            H5S.close(MSID)
            H5T.close(TID)
            H5D.close(DID)
            
            %% Get the normals.
            normals = [...
                dataVertex.NormalX, ...
                dataVertex.NormalY, ...
                dataVertex.NormalZ];                
        end % GetNormals
        
        function pos = GetPositions(obj)
            % GetPositions Get the xyz positions of the surfaces
            %
            %   pos = obj.GetPositions returns an mx3 array of surface
            %   centroids.
            
            %% If the object doesn't have calculated statistics, return.
            if isempty(obj.GIDS8)
                pos = [];
                return
            end % if
            
            %% Read the type data.
            % Read the StatisticsType HDF5 data.
            DID = H5D.open(obj.GIDS8, 'StatisticsType');
            dataStatisticsType = H5D.read(DID);
            H5D.close(DID)
            
            % Organize the stat names.
            typeName = dataStatisticsType.Name';
            typeName = num2cell(typeName, 2);
            typeName = deblank(typeName);
            
            typeID = dataStatisticsType.ID;
            
            %% Mask for position data.
            maskNamePos = ~cellfun(@isempty, ...
                regexp(typeName, '^Position (X|Y|Z)$', 'Match', 'Once'));
            
            typeIDPos = typeID(maskNamePos);
            
            %% Read all the statistic values, then get the position data.
            DID = H5D.open(obj.GIDS8, 'StatisticsValue');
            dataStatisticsValue = H5D.read(DID);
            H5D.close(DID)
            
            xMask = dataStatisticsValue.ID_StatisticsType == typeIDPos(1);
            xIDs = dataStatisticsValue.ID_Object(xMask);
            [~, xOrder] = sort(xIDs);
            xValues = dataStatisticsValue.Value(xMask);
            pos(:, 1) = xValues(xOrder);
            
            yMask = dataStatisticsValue.ID_StatisticsType == typeIDPos(2);
            yIDs = dataStatisticsValue.ID_Object(yMask);
            [~, yOrder] = sort(yIDs);
            yValues = dataStatisticsValue.Value(yMask);
            pos(:, 2) = yValues(yOrder);
            
            zMask = dataStatisticsValue.ID_StatisticsType == typeIDPos(3);
            zIDs = dataStatisticsValue.ID_Object(zMask);
            [~, zOrder] = sort(zIDs);
            zValues = dataStatisticsValue.Value(zMask);
            pos(:, 3) = zValues(zOrder);
        end % GetPositions
        
        function triangles = GetTriangles(obj, sIdx)
            % GetTriangles Get the triangles for a surface object
            %
            %   triangles = obj.GetTriangles(sIdx) returns an mx3 array of
            %   triangles (faces) for the surface represented by the
            %   zero-based index sIdx.
            
            %% Get the number of triangles and vertices dataset.
            DID = H5D.open(obj.GID, 'TimeNVerticesNTriangles');
            dataTimeNVerticesNTriangles = transpose(H5D.read(DID));
            H5D.close(DID)
            
            %% Calculate the offset and the slab to read for the surface.
            offset = [sum(dataTimeNVerticesNTriangles(1:sIdx, 3), 1), 0];
            slab = [double(dataTimeNVerticesNTriangles(sIdx + 1, 3)), 3];
            
            %% Read the triangle slab for the surface.
            DID = H5D.open(obj.GID, 'Triangles');
            FSID = H5D.get_space(DID);
            MSID = H5S.create_simple(2, slab, []);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);            
            triangles = transpose(H5D.read(...
                DID, 'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT'));
            
            %% Close the HDF5 objects.
            H5S.close(MSID)
            H5S.close(FSID)
            H5D.close(DID)
        end % GetTriangles
        
        function vertices = GetVertices(obj, sIdx)
            % GetVertices Get the vertices for a surface object
            %
            %   vertices = obj.GetVertices(sIdx) returns an mx3 array of
            %   surface vertices for the surface represented by the
            %   zero-based index sIdx.
            
            %% Get the number of triangles and vertices.
            DID = H5D.open(obj.GID, 'TimeNVerticesNTriangles');
            dataTimeNVerticesNTriangles = transpose(H5D.read(DID));
            H5D.close(DID)
            
            %% Calculate the offest and slab to read.
            offset = [sum(dataTimeNVerticesNTriangles(1:sIdx, 2), 1), 0];
            slab = [double(dataTimeNVerticesNTriangles(sIdx + 1, 2)), 3];
            
            %% Create the HDF5 objects to access the slab.
            DID = H5D.open(obj.GID, 'Vertices');
            FSID = H5D.get_space(DID);
            MSID = H5S.create_simple(2, slab, []);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);            
            
            %% Read the vertices.
            vertices = transpose(H5D.read(...
                DID, 'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT'));
            
            %% Close the HDF5 objects.
            H5S.close(MSID)
            H5S.close(FSID)
            H5D.close(DID)
        end % GetVertices
        
    end % methods
    
    events 
        
    end % events
    
end % class SurfacesReader
