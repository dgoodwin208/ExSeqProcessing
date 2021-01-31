classdef CellsReader < SurpassObjectReader
    % CellsReader Read Imaris ims file Cells object data
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader | SurpassObjectReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        NumberOfCells % Number of cells in the object
        NumberOfNuclei % Number of nuclei
        NumberOfVesicleTypes % Number of vesicle types in the object
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    properties (SetAccess = 'immutable', GetAccess = 'private', Hidden = true, Transient = true)
        
        MasksGID % HDF5 group ID for the object cell masks data 
        MasksDataSetGID % HDF5 group ID for the masks DataSet
        MasksGIDS8 % HDF5 group ID for the masks stats
        
    end % properties (SetAccess = 'private', GetAccess = 'private', Transient = true)
    
    methods
        
        function obj = CellsReader(GID)
            % CellsReader Create a Cells reader object
            %
            % obj = CellsReader(GID) constructs obj to read Imaris Cells
            % data. GID is an HDF5 Group Identifier for a /Cells group in
            % an ims file.
            
            %% Call the superclass constructor.
            obj = obj@SurpassObjectReader(GID, 'Cells');
            
            %% Check for the required data set for a valid Cells object.
            cellsGID = H5G.open(GID, 'Cells');
            nucleiGID = H5G.open(GID, 'Nuclei');
            vesiclesGID = H5G.open(GID, 'Vesicles');
            
            isValidCells = ...
                H5L.exists(GID, 'CellNNucleiNVesicles', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'NucleiVesiclesIds', 'H5P_DEFAULT') && ...
                H5L.exists(cellsGID, 'TimeId', 'H5P_DEFAULT') && ...
                H5L.exists(cellsGID, 'TimeInfos', 'H5P_DEFAULT') && ...
                H5L.exists(nucleiGID, 'TimeId', 'H5P_DEFAULT') && ...
                H5L.exists(nucleiGID, 'TimeInfos', 'H5P_DEFAULT') && ...
                H5L.exists(vesiclesGID, 'CoordsXYZR', 'H5P_DEFAULT') && ...
                H5L.exists(vesiclesGID, 'Time', 'H5P_DEFAULT') && ...
                H5L.exists(vesiclesGID, 'TimeInfos', 'H5P_DEFAULT');
            
            H5G.close(cellsGID)
            H5G.close(nucleiGID)
            H5G.close(vesiclesGID)
                        
            if ~isValidCells
                return
            end % if
            
            %% Get the cell masks group.
            % Get the masks ID.
            aID = H5A.open(obj.GID, 'ImageMasksId');
            idMasks = str2double(H5A.read(aID));
            H5A.close(aID)
            
            % Find the matching ImageMasks group.
            groupName = H5I.get_name(GID);
            stringCellsNumber = regexp(groupName, '\d{1,}$', 'Match', 'Once');
            
            if H5L.exists(GID, ['/Scene/Content/ImageMasks' stringCellsNumber], 'H5P_DEFAULT')
                obj.MasksGID = H5G.open(GID, ...
                    ['/Scene/Content/ImageMasks' stringCellsNumber]);
                
                % Compare the Imaris Ids.
                aID = H5A.open(obj.MasksGID, 'ImageMasksId');
                idImageMasks = str2double(H5A.read(aID));
                H5A.close(aID)
                
                if idImageMasks == idMasks
                    %% Open the masks DataSet group.
                    aID = H5A.open(obj.MasksGID, 'Id');
                    idMasksDataSet = H5A.read(aID);
                    H5A.close(aID)
                    
                    obj.MasksDataSetGID = H5G.open(GID, ...
                        ['/DataSet' num2str(idMasksDataSet)]);
                    
                    %% Open the masks Scene8 group.
                    obj.MasksGIDS8 = H5G.open(GID, ...
                        ['/Scene8/Content/ImageMasks' num2str(stringCellsNumber)]);
                    
                    %% Get the number of cells, nuclei and vesicle types.
                    DID = H5D.open(GID, 'CellNNucleiNVesicles');
                    cellNNucleiNVesicles = transpose(H5D.read(DID));
                    obj.NumberOfCells = size(cellNNucleiNVesicles, 1);
                    obj.NumberOfNuclei = cellNNucleiNVesicles(:, 2);
                    H5D.close(DID)

                    aID = H5A.open(GID, 'NumberOfVesicleTypes');
                    obj.NumberOfVesicleTypes = H5A.read(aID);
                    H5A.close(aID)
                end % if
            end % if
        end % CellsReader
        
        function delete(obj)
            % delete Destructor function for CellsReader objects
            %
            %
            
            %% Close the Cells-specific groups.
            H5G.close(obj.MasksGIDS8)
            H5G.close(obj.MasksDataSetGID)
            H5G.close(obj.MasksGID)
        end % delete
        
        function mask = GetCell(obj, cIdx)
            % GetCell Get a mask for a cell 
            % 
            %   mask = obj.GetCell(cIdx) returns a binary mask for the cell
            %   reprented by the zero-based index cIdx.
            
            %% Read the Cell data in /Scene8. 
            DID = H5D.open(obj.GIDS8, 'Cells/Cell');
            dataCell = H5D.read(DID);
            H5D.close(DID)
            
            %% Get the time index and mask ID for the cell.
            tIdx = dataCell.IDMaskImageTime(cIdx + 1);
            mIdx = dataCell.IDMaskImageIndex(cIdx + 1);
            
            %% Get the colors assigned to the cell mask.
            DID = H5D.open(obj.MasksGIDS8, 'Label');
            dataLabel = H5D.read(DID);
            H5D.close(DID);
            
            colorValues = uint32(dataLabel.Color(dataLabel.Label == mIdx));
            
            %% Read the labels dataset data.
            DID = H5D.open(obj.MasksDataSetGID, ...
                ['ResolutionLevel 0/TimePoint ' ...
                num2str(tIdx) '/Channel 0/Data']);
            data = H5D.read(DID);
            H5D.close(DID)
            
            %% Mask on the cell label values.
            mask = ismember(data, colorValues);
        end % GetCell
        
        function ids = GetIDs(obj)
            % GetIDs Get the IDs of all cells
            %
            %   ids = obj.GetIDs returns a vector of IDs for all the cells.
            
            %% Open the CellNNucleiNVesicles data set.
            DID = H5D.open(obj.GID, 'CellNNucleiNVesicles');
            dataCellNNucleiNVesicles = H5D.read(DID);
            H5D.close(DID)
            
            %% Get the IDs.
            ids = transpose(dataCellNNucleiNVesicles(1, :));
        end % GetIDs
        
        function idx = GetIndicesT(obj)
            % GetIndicesT Get the time indices of the cells
            %
            %   idx = obj.GetIndicesT returns a vector of zero-based
            %   indices indicating the time index of each cell.

            %% Open the Time dataset.
            DID = H5D.open(obj.GID , 'Cells/TimeId');
            dataTimeId = H5D.read(DID);
            H5D.close(DID)
            
            %% Get the time indices from the data.
            idx = transpose(dataTimeId(1, :));
        end % GetIndicesT
        
        function mask = GetNucleus(obj, nIdx)
            % GetNucleus Get the mask for a nucleus
            %
            %   nucleus = obj.GetNucleus(nIdx) returns a mask for the
            %   nucleus represented by the zero-based index nIdx.
            
            %% Read the Nucleus data in /Scene8. 
            DID = H5D.open(obj.GIDS8, 'Nuclei/Nucleus');
            dataNucleus = H5D.read(DID);
            H5D.close(DID)
            
            %% Get the time index and mask ID for the nucleus.
            tIdx = dataNucleus.IDMaskImageTime(nIdx + 1);
            mIdx = dataNucleus.IDMaskImageIndex(nIdx + 1);
            
            %% Get the colors assigned to the nucleus mask.
            DID = H5D.open(obj.MasksGIDS8, 'Label');
            dataLabel = H5D.read(DID);
            H5D.close(DID);
            
            colorValues = uint32(dataLabel.Color(dataLabel.Label == mIdx));
            
            %% Read the labels dataset data.
            DID = H5D.open(obj.MasksDataSetGID, ...
                ['ResolutionLevel 0/TimePoint ' ...
                num2str(tIdx) '/Channel 0/Data']);
            data = H5D.read(DID);
            H5D.close(DID)
            
            %% Mask on the cell label values.
            mask = ismember(data, colorValues);
        end % GetNucleus
            
        function ids = GetNucleiIDs(obj)
            % GetNucleiIDs Returns the IDs of all nuclei
            %
            %   ids = obj.GetNucleiIDs returns a vector of IDs for all the
            %   nuclei.
            
            %% Open the Nucleus data set in /Scene8.
            DID = H5D.open(obj.GIDS8, 'Nuclei/Nucleus');
            dataNucleus = H5D.read(DID);
            H5D.close(DID)
            
            ids = dataNucleus.ID;
        end % GetNucleiIDs
        
        function idx = GetNucleiIndicesT(obj)
            % GetNucleiIndicesT Get the time indices of the nuclei
            %
            % idx = obj.GetNucleiIndicesT(nIdx) returns a vector of
            % zeros-based indices indicating the time index of each
            % nucleus.
            
            %% Open the Time dataset.
            DID = H5D.open(obj.GID, 'Nuclei/TimeId');
            dataTimeId = H5D.read(DID);
            H5D.close(DID)
            
            %% Close the HDF5 dataset object.
            idx = transpose(dataTimeId(1, :));
        end % GetNucleiIndicesT
        
        function pos = GetNucleiPositions(obj)
            % GetNucleiPositions Get the xyz positions of the nuclei
            %
            %   pos = obj.GetNucleiPositions returns an mx3 array of nuclei
            %   centroids.
            
            %% If the object doesn't have calculated statistics, return.
            if isempty(obj.GIDS8)
                pos = [];
                return
            end % if
            
            %% Read the type data.
            % Read the StatisticsType HDF5 data.
            datasetID = H5D.open(obj.GIDS8, 'StatisticsType');
            dataStatisticsType = H5D.read(datasetID);
            H5D.close(datasetID)
            
            % Organize the stat names.
            typeName = dataStatisticsType.Name';
            typeName = num2cell(typeName, 2);
            typeName = deblank(typeName);
            
            typeID = dataStatisticsType.ID;
            
            %% Mask for position data.
            maskPosX = ~cellfun(@isempty, ...
                regexp(typeName, '^Nucleus Position X$', 'Match', 'Once'));
            idPosX = typeID(maskPosX);
            
            maskPosY = ~cellfun(@isempty, ...
                regexp(typeName, '^Nucleus Position Y$', 'Match', 'Once'));
            idPosY = typeID(maskPosY);

            maskPosZ = ~cellfun(@isempty, ...
                regexp(typeName, '^Nucleus Position Z$', 'Match', 'Once'));
            idPosZ = typeID(maskPosZ);

            %% Read all the statistic values, then get the position data.
            datasetID = H5D.open(obj.GIDS8, 'StatisticsValue');
            dataStatisticsValue = H5D.read(datasetID);
            H5D.close(datasetID)
            
            xMask = dataStatisticsValue.ID_StatisticsType == idPosX(1);
            xIDs = dataStatisticsValue.ID_Object(xMask);
            [~, xOrder] = sort(xIDs);
            xValues = dataStatisticsValue.Value(xMask);
            pos(:, 1) = xValues(xOrder);
            
            yMask = dataStatisticsValue.ID_StatisticsType == idPosY(2);
            yIDs = dataStatisticsValue.ID_Object(yMask);
            [~, yOrder] = sort(yIDs);
            yValues = dataStatisticsValue.Value(yMask);
            pos(:, 2) = yValues(yOrder);
            
            zMask = dataStatisticsValue.ID_StatisticsType == idPosZ(3);
            zIDs = dataStatisticsValue.ID_Object(zMask);
            [~, zOrder] = sort(zIDs);
            zValues = dataStatisticsValue.Value(zMask);
            pos(:, 3) = zValues(zOrder);
        end % GetNucleiPositions
        
        function edges = GetNucleiTrackEdges(obj)
            % GetNucleiTrackEdges Get nuclei track edges
            %
            %   edges = obj.GetNucleiTrackEdges returns an mx4 array
            %   containing the track connections for the nuclei. Each row
            %   of edges represents a connection between a cell and
            %   associated nucleus in the format:
            %
            %       [cellA, nucleusA, cellB, nucleusB]
            
            %% Read the generic vesicles edges array.
            DID = H5D.open(obj.GID, 'Nuclei/Edges');
            dataEdges = transpose(H5D.read(DID));
            H5D.close(DID)
            
            %% Map the nuclei edges to the parent cells.
            GID = H5G.open(obj.GIDS8, 'Nuclei');

            if H5L.exists(GID, 'NucleusCellOffset', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'TrackEdge0', 'H5P_DEFAULT')
                %% Read the NucleusCellOffset and TrackEdge0 datasets.
                DID = H5D.open(GID, 'NucleusCellOffset');
                dataNucleusCellOffset = transpose(H5D.read(DID));
                H5D.close(DID)
                
                DID = H5D.open(GID, 'TrackEdge0');
                dataTrackEdge0 = H5D.read(DID);
                H5D.close(DID)
                
                H5G.close(GID)
                
                %% Construct the cell ID list.
                cellIDs = cell2mat(arrayfun(...
                    @(c, a, b)repmat(c, [1, b - a]), ...
                    dataNucleusCellOffset.ID', ...
                    dataNucleusCellOffset.IndexBegin', ...
                    dataNucleusCellOffset.IndexEnd', ...
                    'UniformOutput', 0));

                %% Map the edges to the cells.
                edges = zeros(size(dataTrackEdge0.ID_ObjectA, 1), 4, 'int64');
                
                edges(:, 1) = cellIDs(dataEdges(:, 1) + 1);
                edges(:, 2) = dataTrackEdge0.ID_ObjectA;
                edges(:, 3) = cellIDs(dataEdges(:, 2) + 1);
                edges(:, 4) = dataTrackEdge0.ID_ObjectB;
            
            else
                edges = [];

            end % if
        end % GetNucleiTrackEdges
        
        function ids = GetNucleiTrackIDs(obj)
            % GetNucleiTrackIDs Get the track IDs for the nuclei track
            % edges
            %
            %   ids = obj.GetNucleiTrackIDs returns a vector containing the
            %   track IDs for the nuclei track edges.
            
            %% Create the IDs from the Track0 dataset. 
            if H5L.exists(obj.GIDS8, 'Nuclei/Track0', 'H5P_DEFAULT');
                DID = H5D.open(obj.GIDS8, 'Nuclei/Track0');
                dataTrack0 = H5D.read(DID);
                H5D.close(DID)

                %% Construct the track ID list.
                cellIDs = arrayfun(...
                    @(c, a, b)repmat(c, [1, b - a]), ...
                    dataTrack0.ID', ...
                    dataTrack0.IndexTrackEdgeBegin', ...
                    dataTrack0.IndexTrackEdgeEnd', ...
                    'UniformOutput', 0);

                ids = transpose([cellIDs{:}]);
                
            else
                ids = [];
                
            end % if
        end % GetNucleiTrackIDs
        
        function pos = GetPositions(obj)
            % GetPositions Get the xyz positions of the cells
            %
            %   pos = obj.GetPositions returns an mx3 array of cell
            %   centroids.
            
            %% If the object doesn't have calculated statistics, return.
            if isempty(obj.GIDS8)
                pos = [];
                return
            end % if
            
            %% Read the type data.
            % Read the StatisticsType HDF5 data.
            datasetID = H5D.open(obj.GIDS8, 'StatisticsType');
            dataStatisticsType = H5D.read(datasetID);
            H5D.close(datasetID)
            
            % Organize the stat names.
            typeName = dataStatisticsType.Name';
            typeName = num2cell(typeName, 2);
            typeName = deblank(typeName);
            
            typeID = dataStatisticsType.ID;
            
            %% Mask for position data.
            maskPosX = ~cellfun(@isempty, ...
                regexp(typeName, '^Cell Position X$', 'Match', 'Once'));
            idPosX = typeID(maskPosX);
            
            maskPosY = ~cellfun(@isempty, ...
                regexp(typeName, '^Cell Position Y$', 'Match', 'Once'));
            idPosY = typeID(maskPosY);

            maskPosZ = ~cellfun(@isempty, ...
                regexp(typeName, '^Cell Position Z$', 'Match', 'Once'));
            idPosZ = typeID(maskPosZ);

            %% Read all the statistic values, then get the position data.
            datasetID = H5D.open(obj.GIDS8, 'StatisticsValue');
            dataStatisticsValue = H5D.read(datasetID);
            H5D.close(datasetID)
            
            xMask = dataStatisticsValue.ID_StatisticsType == idPosX(1);
            xIDs = dataStatisticsValue.ID_Object(xMask);
            [~, xOrder] = sort(xIDs);
            xValues = dataStatisticsValue.Value(xMask);
            pos(:, 1) = xValues(xOrder);
            
            yMask = dataStatisticsValue.ID_StatisticsType == idPosY(1);
            yIDs = dataStatisticsValue.ID_Object(yMask);
            [~, yOrder] = sort(yIDs);
            yValues = dataStatisticsValue.Value(yMask);
            pos(:, 2) = yValues(yOrder);
            
            zMask = dataStatisticsValue.ID_StatisticsType == idPosZ(1);
            zIDs = dataStatisticsValue.ID_Object(zMask);
            [~, zOrder] = sort(zIDs);
            zValues = dataStatisticsValue.Value(zMask);
            pos(:, 3) = zValues(zOrder);
        end % GetPositions
        
        function edges = GetTrackEdges(obj)
            % GetTrackEdges Get the edges (connections) between cells
            %
            %   edges = obj.GetTrackEdges returns an mx2 array containing
            %   the track edges (connections) for the cells.
            
            %% Read the generic edges, then map them to the cell IDs.
            if H5L.exists(obj.GIDS8, 'Cells/Track0', 'H5P_DEFAULT')
                %% Read the edges from the Track0 dataset.
                DID = H5D.open(obj.GIDS8, 'Cells/TrackEdge0');
                dataTrackEdge0 = H5D.read(DID);
                H5D.close(DID)
                
                edges = [dataTrackEdge0.ID_ObjectA, dataTrackEdge0.ID_ObjectB];
                
            else
                edges = [];

            end % if
        end % GetTrackEdges
        
        function ids = GetTrackIDs(obj)
            % GetTrackIDs Get the track to which each edge belongs
            %
            %   ids = obj.GetTrackIDs returns a vector containing
            %   the track for each edge (connected pair of objects).
            
            %% Create the IDs from the Track0 dataset. 
            if H5L.exists(obj.GIDS8, 'Cells/Track0', 'H5P_DEFAULT')
                %% Read the Track0 dataset.
                DID = H5D.open(obj.GIDS8, 'Cells/Track0');
                dataTrack0 = H5D.read(DID);
                H5D.close(DID)

                %% Construct the track ID list.
                cellIDs = arrayfun(...
                    @(c, a, b)repmat(c, [1, b - a]), ...
                    dataTrack0.ID', ...
                    dataTrack0.IndexTrackEdgeBegin', ...
                    dataTrack0.IndexTrackEdgeEnd', ...
                    'UniformOutput', 0);

                ids = transpose([cellIDs{:}]);

            else
                ids = [];
                
            end % if
        end % GetTrackIDs
        
        function ids = GetVesiclesIDs(obj, vIdx)
            % GetVesiclesIDs Get the IDs for vesicles of a type
            %
            %	ids = obj.GetVesiclesIDs(vIdx) returns the IDs for all the
            %   vesicles represented by the zero-based index vIdx, which
            %	corresponds to Type A (0), Type B (1) vesicles, etc.
            
            %% Parse the input.
            parser = inputParser;
            parser.addRequired('obj', @(obj)isa(obj, 'CellsReader'))
            parser.addRequired('vIdx', @(vIdx)isscalar(vIdx) && vIdx >= 0)
            
            parse(parser, obj, vIdx)
            
            %% Get the vesicle IDs.
            if H5L.exists(obj.GIDS8, ['Vesicles' num2str(vIdx)], 'H5P_DEFAULT')
                %% Open the Vesicles data set in /Scene8.
                DID = H5D.open(obj.GIDS8, ['Vesicles' num2str(vIdx) '/Vesicle']);
                dataVesicle = H5D.read(DID);
                H5D.close(DID)

                ids = dataVesicle.ID;
                
            else
                ids = [];
                
            end % if
        end % GetVesiclesIDs
        
        function pos = GetVesiclesPositions(obj, cIdx, vIdx)
            % GetVesiclesPositions Get the positions of vesicles in a cell
            %
            %   pos = obj.GetVesiclesPositions(cIdx, vIdx) returns the
            %   vesicle positions for the cell represented by the
            %   zero-based index cIdx and the vesicle type represented by
            %   vIdx, which corresponds to Type A (0), Type B (1) vesicles,
            %   etc.
            
            %% Parse the input.
            parser = inputParser;
            parser.addRequired('obj', @(obj)isa(obj, 'CellsReader'))
            parser.addRequired('cIdx', @(cIdx)isscalar(cIdx) && cIdx >= 0)
            parser.addRequired('vIdx', @(vIdx)isscalar(vIdx) && vIdx >= 0)
            
            parse(parser, obj, cIdx, vIdx)
            
            %% Read the VesicleCellOffset dataset.
            GID = H5G.open(obj.GIDS8, ['Vesicles' num2str(vIdx)]);
            DID = H5D.open(GID, 'VesicleCellOffset');
            dataVesicleCellOffset = transpose(H5D.read(DID));
            H5D.close(DID)
            
            %% Calculate the offset and slab to read.
            rowIdx = dataVesicleCellOffset.ID == cIdx;

            if isempty(dataVesicleCellOffset.IndexBegin(rowIdx))
                pos = [];
                return
            end % if

            offset = double([dataVesicleCellOffset.IndexBegin(rowIdx) 0]);
            slab = double([...
                dataVesicleCellOffset.IndexEnd(rowIdx) - ...
                dataVesicleCellOffset.IndexBegin(rowIdx) ...
                3]);
            
            %% Read the vesicle positions.
            if vIdx == 0
                groupstring = 'Vesicles';
                
            else
                groupstring = ['Vesicles' num2str(vIdx)];
            
            end %if
            
            if H5L.exists(obj.GID, groupstring, 'H5P_DEFAULT')
                DID = H5D.open(obj.GID, [groupstring '/CoordsXYZR']);
                MSID = H5S.create_simple(2, slab, []);
                FSID = H5D.get_space(DID);
                H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);
                pos = transpose(H5D.read(...
                    DID, 'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT'));
                
                % Close the HDF5 objects.
                H5S.close(FSID)
                H5S.close(MSID)
                H5D.close(DID)

            else
                pos = [];

            end % if
        end % GetVesiclesPositions
        
        function radii = GetVesiclesRadii(obj, cIdx, vIdx)
            % GetVesiclesRadii Get the radii of vesicles in a cell
            %
            %   radii = obj.GetVesicleRadii(cIdx, vIdx) returns the
            %   vesicle radii for the cell represented by the zero-based
            %   index cIdx and the vesicle type represented by vIdx, which
            %   corresponds to Type A (0), Type B (1) vesicles, etc.
            
            %% Parse the input.
            parser = inputParser;
            parser.addRequired('obj', @(obj)isa(obj, 'CellsReader'))
            parser.addRequired('cIdx', @(cIdx)isscalar(cIdx) && cIdx >= 0)
            parser.addRequired('vIdx', @(vIdx)isscalar(vIdx) && vIdx >= 0)
            
            parse(parser, obj, cIdx, vIdx)
            
            %% Read the VesicleCellOffset dataset.
            GID = H5G.open(obj.GIDS8, ['Vesicles' num2str(vIdx)]);
            DID = H5D.open(GID, 'VesicleCellOffset');
            dataVesicleCellOffset = transpose(H5D.read(DID));
            H5D.close(DID)
            
            %% Calculate the offset and slab to read.
            rowIdx = dataVesicleCellOffset.ID == cIdx;

            if isempty(dataVesicleCellOffset.IndexEnd(rowIdx))
                radii = [];
                return
            end % if
            
            offset = double([dataVesicleCellOffset.IndexBegin(rowIdx), 3]);
            slab = double([...
                dataVesicleCellOffset.IndexEnd(rowIdx) - ...
                dataVesicleCellOffset.IndexBegin(rowIdx) ...
                1]);
            
            %% Read the vesicle positions.
            if vIdx == 0
                groupstring = 'Vesicles';
                
            else
                groupstring = ['Vesicles' num2str(vIdx)];
            
            end %if
            
            if H5L.exists(obj.GID, groupstring, 'H5P_DEFAULT')
                DID = H5D.open(obj.GID, [groupstring '/CoordsXYZR']);
                MSID = H5S.create_simple(2, slab, []);
                FSID = H5D.get_space(DID);
                H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], slab);
                radii = transpose(H5D.read(...
                    DID, 'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT'));
                
                % Close the HDF5 objects.
                H5S.close(FSID)
                H5S.close(MSID)
                H5D.close(DID)

            else
                radii = [];

            end % if
        end % GetVesiclesRadii
        
        function edges = GetVesiclesTrackEdges(obj, vIdx)
            % GetVesiclesTrackEdges Get vesicle track edges
            %
            %   edges = obj.GetVesiclesTrackEdges(vIdx) returns an mx4
            %   array containing the track connections for the vesicles
            %   represented by vIdx, which corresponds to Type A (0), Type
            %   B (1) vesicles, etc. Each row of edges represents a
            %   connection between a cell and associated vesicle in the
            %   format:
            %
            %       [cellA, vesicleA, cellB, vesicleB]
            
            %% Parse the input.
            parser = inputParser;
            parser.addRequired('obj', @(obj)isa(obj, 'CellsReader'))
            parser.addRequired('vIdx', @(vIdx)isscalar(vIdx) && vIdx >= 0)
            
            parse(parser, obj, vIdx)
            
            %% Read the vesicles edges array.
            if vIdx == 0
                groupString = 'Vesicles';
                
            else
                groupString = ['Vesicles' num2str(vIdx)];
                
            end % if
            
            DID = H5D.open(obj.GID, [groupString '/Edges']);
            dataEdges = transpose(H5D.read(DID));
            H5D.close(DID)

            %% Map the vesicle edges to the parent cells.
            GID = H5G.open(obj.GIDS8, ['Vesicles' num2str(vIdx)]);

            if H5L.exists(GID, 'VesicleCellOffset', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'TrackEdge0', 'H5P_DEFAULT')
                %% Read the VesicleCellOffset and TrackEdge0 datasets.
                DID = H5D.open(GID, 'VesicleCellOffset');
                dataVesicleCellOffset = transpose(H5D.read(DID));
                H5D.close(DID)
                
                DID = H5D.open(GID, 'TrackEdge0');
                dataTrackEdge0 = H5D.read(DID);
                H5D.close(DID)
                
                H5G.close(GID)
                
                %% Construct the cell ID list.
                cellIDs = cell2mat(arrayfun(...
                    @(c, a, b)repmat(c, [1, b - a]), ...
                    dataVesicleCellOffset.ID', ...
                    dataVesicleCellOffset.IndexBegin', ...
                    dataVesicleCellOffset.IndexEnd', ...
                    'UniformOutput', 0));

                %% Map the edges to cells.
                edges = zeros(size(dataTrackEdge0.ID_ObjectA, 1), 4, 'int64');
                
                edges(:, 1) = cellIDs(dataEdges(:, 1) + 1);
                edges(:, 2) = dataTrackEdge0.ID_ObjectA;
                edges(:, 3) = cellIDs(dataEdges(:, 2) + 1);
                edges(:, 4) = dataTrackEdge0.ID_ObjectB;
            
            else
                edges = [];

            end % if
        end % GetVesiclesTrackEdges
        
        function ids = GetVesiclesTrackIDs(obj, vIdx)
            % GetVesiclesTrackIDs Get the track IDs for the vesicles' track
            % edges
            %
            %   ids = obj.GetVesiclesTrackIDs(vIdx) returns vector
            %   containing the track IDs for the vesicles' track edges for
            %   the vesicle type represented by vIdx, which corresponds to
            %   Type A (0), Type B (1) vesicles, etc.
            
            %% Parse the input.
            parser = inputParser;
            parser.addRequired('obj', @(obj)isa(obj, 'CellsReader'))
            parser.addRequired('vIdx', @(vIdx)isscalar(vIdx) && vIdx >= 0)
            
            parse(parser, obj, vIdx)
            
            %% Create the IDs from the Track0 dataset. 
            if H5L.exists(obj.GIDS8, ['Vesicles' num2str(vIdx) '/Track0'], 'H5P_DEFAULT');
                DID = H5D.open(obj.GIDS8, ['Vesicles' num2str(vIdx) '/Track0']);
                dataTrack0 = H5D.read(DID);
                H5D.close(DID)

                %% Construct the track ID list.
                cellIDs = arrayfun(...
                    @(c, a, b)repmat(c, [1, b - a]), ...
                    dataTrack0.ID', ...
                    dataTrack0.IndexTrackEdgeBegin', ...
                    dataTrack0.IndexTrackEdgeEnd', ...
                    'UniformOutput', 0);

                ids = transpose([cellIDs{:}]);
                
            else
                ids = [];
                
            end % if
        end % GetVesicleTrackIDs

    end % methods
    
end % class CellsReader
