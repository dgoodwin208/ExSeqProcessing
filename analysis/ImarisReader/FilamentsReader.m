classdef FilamentsReader < SurpassObjectReader
    % FilamentsReader Read Imaris ims file Filaments object data
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader | SurpassObjectReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        NumberOfFilaments % Number of filaments in the Filaments object
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    methods
        
        function obj = FilamentsReader(GID)
            % FilamentsReader Create a Filaments reader object
            %
            % obj = FilamentsReader(GID) constructs obj
            % to read Imaris Filaments data. GID is an HDF5 Group
            % Identifier for a /Filaments group in an ims file.
            
            %% Call the superclass constructor.
            obj = obj@SurpassObjectReader(GID, 'Filaments');
            
            %% Check for the required data sets for a valid Filaments object.
            graphsGID = H5G.open(GID, 'Graphs');
            graphtracksGID = H5G.open(GID, 'GraphTracks');
            
            isValidFilaments = ...
                H5L.exists(graphsGID, 'Segments', 'H5P_DEFAULT') && ...
                H5L.exists(graphsGID, 'TimeInfos', 'H5P_DEFAULT') && ...
                H5L.exists(graphsGID, 'TimesNVerticesNEdgesRoots', 'H5P_DEFAULT') && ...
                H5L.exists(graphsGID, 'Vertices', 'H5P_DEFAULT') && ...
                H5L.exists(graphtracksGID, 'GraphVertex', 'H5P_DEFAULT') && ...
                H5L.exists(graphtracksGID, 'TimeInfos', 'H5P_DEFAULT');
                
            %% Get the total number of filaments.
            if isValidFilaments
                DID = H5D.open(graphsGID, 'TimesNVerticesNEdgesRoots');
                FSID = H5D.get_space(DID);
                [~, spaceDims, ~] = H5S.get_simple_extent_dims(FSID);
                obj.NumberOfFilaments = spaceDims(1);

                % Close the HDF5 dataset objects.
                H5S.close(FSID);
                H5D.close(DID)
            end % if
            
            % Close the graphs Group.
            H5G.close(graphsGID)
            H5G.close(graphtracksGID)
        end % FilamentsReader
        
        function idx = GetBeginningVertexIndex(obj, fIdx)
            % GetBeginningVertex Get the beginning vertex of the filaments
            % 
            %   idx = obj.GetBeginningVertex(fIdx) returns the index of the
            %   initial vertex for all the filament represented by the
            %   zero-based index fIdx.
                        
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Get the initial index.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            idx = dataTimesNVerticesNEdgesRoots(4, order == fIdx + 1);
        end % GetBeginningVertex
        
        function edges = GetEdges(obj, fIdx)
            % GetEdges Get the edges between vertices for a filament
            %
            %   edges = obj.GetEdges(fIdx) returns the edges between
            %   vertices for the filament represented by the zero-based
            %   index fIdx.
            
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Sort the data.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            nEdges = dataTimesNVerticesNEdgesRoots(3, order);
            
            %% Calculate the slab to read and the offset position.
            offset = sum(nEdges(1:fIdx)) + 1;
            slab = double(nEdges(fIdx + 1)) - 1;
            
            %% Read the Segments data.
            DID = H5D.open(obj.GID, 'Graphs/Segments');
            dataSegments = H5D.read(DID);
            H5D.close(DID)
            
            %% Get the edges.
            edges = transpose(dataSegments(:, offset:offset + slab));
        end % GetEdges
        
        function ids = GetEdgesSegmentID(obj, fIdx)
            % GetEdgesSegmentID Get the ids for edges of segments
            %
            %   ids = obj.GetEdgesSegmentID(fIdx) returns the segment to
            %   which each edge belongs for the filament represented by
            %   fIdx. Each element in ids corresponds to an edge returned
            %   by the GetEdges method.
            
            %% Read the filament dataset and get the indices to return.
            DID = H5D.open(obj.GIDS8, 'Filament');
            dataFilament = H5D.read(DID);
            H5D.close(DID)

            fIdxs = ...
                dataFilament.IndexEdgeBegin(fIdx + 1) + 1:...
                dataFilament.IndexEdgeEnd(fIdx + 1);

            %% Create the dendrite ID list.
            DID = H5D.open(obj.GIDS8, 'DendriteSegment');
            dataDendriteSegment = H5D.read(DID);
            H5D.close(DID)
            
            idDendrites = cell2mat(arrayfun(...
                @(c, a, b)repmat(c, [1, b - a]), ...
                dataDendriteSegment.ID', ...
                dataDendriteSegment.IndexDendriteSegmentEdgeBegin', ...
                dataDendriteSegment.IndexDendriteSegmentEdgeEnd', ...
                'UniformOutput', 0))';
            
            %% Create the spine ID list.
            DID = H5D.open(obj.GIDS8, 'Spine');
            dataSpine = H5D.read(DID);
            H5D.close(DID)
            
            idSpines = cell2mat(arrayfun(...
                @(c, a, b)repmat(c, [1, b - a]), ...
                dataSpine.ID', ...
                dataSpine.IndexSpineEdgeBegin', ...
                dataSpine.IndexSpineEdgeEnd', ...
                'UniformOutput', 0))';
            
            %% Read the edge datasets and create the edge order.
            DID = H5D.open(obj.GIDS8, 'DendriteSegmentEdge');
            dataDendriteSegmentEdge = H5D.read(DID);
            H5D.close(DID)

            DID = H5D.open(obj.GIDS8, 'SpineEdge');
            dataSpineEdge = H5D.read(DID);
            H5D.close(DID)
            
            [~, order] = sort([dataDendriteSegmentEdge.IndexEdge; ...
                dataSpineEdge.IndexEdge]);
            
            %% Get the edges for the indicated filament.
            idAll = [idDendrites; idSpines];
            idAll = idAll(order);
            
            ids = idAll(fIdxs);
        end % GetEdgesSegmentID
        
        function ids = GetIDs(obj)
            % GetIDs Get the IDs for all filaments
            %
            %   ids = obj.GetIDs returns a vector containing the ID for
            %   every filament in the Filaments object.
            
            %% Read the IDs from the Filament data set.
            DID = H5D.open(obj.GIDS8, 'Filament');
            FSID = H5D.get_space(DID);
            TID = H5T.create('H5T_COMPOUND', 8');
            H5T.insert(TID, 'ID', 0, 'H5T_STD_I64LE');
            
            dataFilament = H5D.read(DID, TID, 'H5S_ALL', FSID, 'H5P_DEFAULT');
            ids = dataFilament.ID;
            
            %% Close the HDF5 objects.
            H5T.close(TID)
            H5S.close(FSID)
            H5D.close(DID)            
        end % GetIDs
        
        function idx = GetIndicesT(obj)
            % GetIndicesT Return the time indices of all filaments
            %
            %   idx = obj.GetIndicesT returns a vector containing the
            %   zero-based time index for every filament in the Filaments
            %   object.

            %% Read the Filament data set.
            DID = H5D.open(obj.GIDS8, 'Filament');
            FSID = H5D.get_space(DID);
            TID = H5T.create('H5T_COMPOUND', 8');
            H5T.insert(TID, 'ID_Time', 0, 'H5T_STD_I64LE');
            
            dataIDs = H5D.read(DID, TID, 'H5S_ALL', FSID, 'H5P_DEFAULT');
            idx = dataIDs.ID_Time;
            
            %% Close the HDF5 objects.
            H5T.close(TID)
            H5S.close(FSID)
            H5D.close(DID)            
        end % GetIndicesT
        
        function number = GetNumberOfEdges(obj)
            % GetNumberOfEdges Get the number of edges for each filament
            %
            %   num = obj.GetNumberOfEdges returns a vector containing the
            %   number of edges for each filament in the Filaments
            %   object.
            
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Sort the data.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            number = transpose(dataTimesNVerticesNEdgesRoots(3, order));
        end % GetNumberOfEdges
        
        function number = GetNumberOfPoints(obj)
            % GetNumberOfPoints Get the number of points for each filament
            %
            %   num = obj.GetNumberOfPoints returns a vector containing the
            %   number of points (vertices) for each filament in the
            %   Filaments object.
            
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Sort the data.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            number = transpose(dataTimesNVerticesNEdgesRoots(2, order));    
        end % GetNumberOfPoints
        
        function pos = GetPositions(obj, fIdx)
            % GetPositions Get the xyz vertex positions of the graph
            %
            %   pos = obj.GetPositions(fIdx) returns an mx3 array of vertex
            %   positions of the graph for the filament object represented
            %   by the zero-based index fIdx.
            
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Sort the data.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            nVertices = dataTimesNVerticesNEdgesRoots(2, order);
            
            %% Calculate the slab to read and the offset position.
            offset = sum(nVertices(1:fIdx));
            slab = double(nVertices(fIdx + 1));
            
            %% Read the Vertex dataset.
            DID = H5D.open(obj.GIDS8, 'Vertex');

            FSID = H5D.get_space(DID);
            MSID = H5S.create_simple(1, slab, []);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);            
            
            TID = H5T.create('H5T_COMPOUND', 24);
            H5T.insert(TID, 'PositionX', 0, 'H5T_NATIVE_DOUBLE')
            H5T.insert(TID, 'PositionY', 8, 'H5T_NATIVE_DOUBLE')
            H5T.insert(TID, 'PositionZ', 16, 'H5T_NATIVE_DOUBLE')
            
            dataVertex = H5D.read(DID, TID, MSID, FSID, 'H5P_DEFAULT');
            
            %% Close the HDF5 objects.
            H5S.close(FSID)
            H5S.close(MSID)
            H5T.close(TID)
            H5D.close(DID)
            
            %%
            pos = [...
                dataVertex.PositionX, ...
                dataVertex.PositionY, ...
                dataVertex.PositionZ];
        end % GetPositions
        
        function radii = GetRadii(obj, fIdx)
            % GetRadii Get the radii of the vertices for a filament
            %
            %   radii = obj.GetRadii returns a vector of filament radii
            
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Sort the data.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            nVertices = dataTimesNVerticesNEdgesRoots(2, order);
            
            %% Calculate the slab to read and the offset position.
            offset = sum(nVertices(1:fIdx));
            slab = double(nVertices(fIdx + 1));
            
            %% Read the Vertex dataset.
            DID = H5D.open(obj.GIDS8, 'Vertex');

            FSID = H5D.get_space(DID);
            MSID = H5S.create_simple(1, slab, []);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);            
            
            TID = H5T.create('H5T_COMPOUND', 8);
            H5T.insert(TID, 'Radius', 0, 'H5T_NATIVE_DOUBLE')
            
            dataVertex = H5D.read(DID, TID, MSID, FSID, 'H5P_DEFAULT');
            
            %% Close the HDF5 objects.
            H5S.close(FSID)
            H5S.close(MSID)
            H5T.close(TID)
            H5D.close(DID)
            
            %%
            radii = dataVertex.Radius;
        end % GetRadii
        
        function edges = GetTrackEdges(obj)
            % GetTrackEdges Get the connections between filaments
            %
            %   edges = obj.GetTrackEdges returns an mx2 array of
            %   edges (track connections) for the filaments.
            
            %% Check for a track edges dataset, then read the edges.
            if H5L.exists(obj.GIDS8, 'TrackEdge0', 'H5P_DEFAULT')
                %% Read the edges.
                DID = H5D.open(obj.GIDS8, 'TrackEdge0');
                dataTrackEdge0 = H5D.read(DID);
                H5D.close(DID)
                
                %% Read the IDs from the Filament data set.
                DID = H5D.open(obj.GIDS8, 'Filament');
                FSID = H5D.get_space(DID);
                TID = H5T.create('H5T_COMPOUND', 8');
                H5T.insert(TID, 'ID', 0, 'H5T_STD_I64LE');

                dataFilament = H5D.read(DID, TID, 'H5S_ALL', FSID, 'H5P_DEFAULT');
            
                %% Close the HDF5 objects.
                H5T.close(TID)
                H5S.close(FSID)
                H5D.close(DID)            
                
                %% Map the IDs to edges.
                mapEdges = containers.Map(...
                    dataFilament.ID, ...
                    int64(0:numel(dataFilament.ID) - 1));

                edgeIDs = [...
                    dataTrackEdge0.ID_ObjectA, ...
                    dataTrackEdge0.ID_ObjectB];
                
                edges = cell2mat(values(mapEdges, num2cell(edgeIDs)));
                
            else
                edges = [];
            
            end % if
        end % GetTrackEdges
        
        function types = GetTypes(obj, fIdx)
            % GetTypes Get the vertices types
            %
            %   types = obj.GetTypes(fIdx) returns a vector of types for
            %   all vertices for the filament represented by the zero-based
            %   index fIdx. The types vector values are 0 for Dendrites and
            %   1 for Spines.
            
            %% Read the TimesNVerticesNEdgesRoots data.
            DID = H5D.open(obj.GID, 'Graphs/TimesNVerticesNEdgesRoots');
            dataTimesNVerticesNEdgesRoots = H5D.read(DID);
            H5D.close(DID)

            %% Sort the data.
            [~, order] = sort(dataTimesNVerticesNEdgesRoots(1, :));
            nVertices = dataTimesNVerticesNEdgesRoots(2, order);
            
            %% Calculate the slab to read and the offset position.
            offset = [sum(nVertices(1:fIdx)), 4];
            slab = [double(nVertices(fIdx + 1)), 1];
            
            %% Create the HDF5 objects to access the slab.
            DID = H5D.open(obj.GID, 'Graphs/Vertices');
            MSID = H5S.create_simple(2, slab, []);
            FSID = H5D.get_space(DID);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);            
            
            %% Read the Vertex dataset.
            types = transpose(H5D.read(...
                DID, 'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT'));
            
            %% Close the HDF5 objects.
            H5S.close(FSID)
            H5S.close(MSID)
            H5D.close(DID)
        end % GetTypes
        
        function edges = GetVertexTrackEdges(obj)
            % GetVertexTrackEdges Get the edges between filament vertices
            %
            %   edges = obj.GetVertexTrackEdges returns an mx4 array of
            %   edges representing track connections for the filament
            %   vertices. Each row of the returned array represents
            %   [filamentA vertexA filamentB vertexB].

            %% Read the Edges and GraphVertex datasets in GraphTracks.
            DID = H5D.open(obj.GID, 'GraphTracks/Edges');
            dataEdges = H5D.read(DID);
            H5D.close(DID)
            
            DID = H5D.open(obj.GID, 'GraphTracks/GraphVertex');
            dataGraphVertex = H5D.read(DID);
            H5D.close(DID)
            
            %% Use the Edges to index the GraphVertex data.
            edges = transpose(reshape(...
                dataGraphVertex(:, dataEdges(:) + 1), ...
                4, []));
            
            %% Sort by the first row of the GraphTracks Edges.
            [~, order] = sort(dataEdges(1, :));
            edges = edges(order, :);
        end % GetVertexTrackEdges
        
        function  ids = GetVertexTrackIDs(obj)
            % GetVertexTrackIDs Get the track for each vertex edge
            %
            %   ids = obj.GetVertexTrackIDs returns the track IDs for each
            %   edge returned by the GetVertexTrackEdges method.
            
            %% Read the Track1 dataset.
            DID = H5D.open(obj.GIDS8, 'Points/Track1');
            dataTrack1 = H5D.read(DID);
            H5D.close(DID)
            
            %% Create the ids list.
            ids = cell2mat(arrayfun(...
                @(c, a, b)repmat(c, [1, b - a]), ...
                dataTrack1.ID', ...
                dataTrack1.IndexTrackEdgeBegin', ...
                dataTrack1.IndexTrackEdgeEnd', ...
                'UniformOutput', 0))';
        end % GetVertexTrackIDs
        
    end % methods
    
end % class FilamentsReader
