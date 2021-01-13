classdef (Abstract) SurpassObjectReader < matlab.mixin.SetGet & dynamicprops
    % SurpassObjectReader Surpass object reader base class
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also CellsReader | FilamentsReader | SpotsReader | SurfacesReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        CreationParameters % Parameters used to create the objects in Imaris
        GID % HDF5 file Group ID for the object data
        Name % Surpass scene object name
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    properties (SetAccess = 'immutable', GetAccess = 'protected', Hidden = true, Transient = true)
        
        GIDS8 % HDF5 Group ID for the object data in /Scene8. 
        
    end % properties (SetAccess = 'private', GetAccess = 'private', Transient = true)
    
    methods
        
        function obj = SurpassObjectReader(GID, surpasstype)
            % SurpassObjectReader Constructor for .ims object reader
            % superclass
            %
            %   SurpassObjectReader is abstract. Instances cannot be
            %   created.
            
            %% Record the Group ID.
            obj.GID = GID;
            
            %% Get the name of the object.
            switch surpasstype

                case 'Cells'
                    % Open the Cells subgroup.
                    cellsGID = H5O.open(GID, 'Cells', 'H5P_DEFAULT');

                    % Record the Name.
                    aID = H5A.open(cellsGID, 'Name');
                    aName = transpose(H5A.read(aID));
                    gLabelIdx = regexp(aName, ' Cell Body$', 'Start', 'Once');
                    obj.Name = aName(1:gLabelIdx);
                    H5A.close(aID)
                    H5G.close(cellsGID)
                    
                case 'Filaments'
                    % Open the GraphTracks subgroup.
                    graphtracksGID = H5O.open(GID, 'GraphTracks', 'H5P_DEFAULT');

                    % Record the Name.
                    aID = H5A.open(graphtracksGID, 'Name');
                    obj.Name = transpose(H5A.read(aID));
                    H5A.close(aID)
                    H5G.close(graphtracksGID)

                case {'Spots', 'Surfaces'}
                    % Record the Name.
                    aID = H5A.open(GID, 'Name');
                    obj.Name = transpose(H5A.read(aID));
                    H5A.close(aID)
                    
            end % switch
            
            %% Get the Imaris identifier for the group in /Scene. 
            aID = H5A.open(GID, 'Id');
            valueId = H5A.read(aID);
            H5A.close(aID)
            
            %% Find the group with the matching identifier in the /Scene8 group.
            % Stats and annotations for each groups are stored in the
            % /Scene8 HDF5 group. Sometimes this group does not exist. If
            % the file doesn't have the group, return.
            if H5L.exists(GID, '/Scene8/Content', 'H5P_DEFAULT')
                scene8GID = H5G.open(GID, '/Scene8/Content');
                
                % Assume that the object's index in /Scene might not match
                % the object's index in /Scene8, and that the scene names
                % are not necessarily unique. Therefore, match the groups
                % using the Imaris Identifier.
                [isMatch, idxMatch, ~] = H5L.iterate(scene8GID, ...
                    'H5_INDEX_NAME', ...
                    'H5_ITER_NATIVE', ...
                    0, ...
                    @matchimarisids, ...
                    valueId);
                
                % If found, open the matching group in /Scene8.
                if isMatch
                    obj.GIDS8 = H5O.open_by_idx(scene8GID, '/Scene8/Content', ...
                        'H5_INDEX_NAME', ...
                        'H5_ITER_INC', ...
                        idxMatch - 1, ...
                        'H5P_DEFAULT');

                    % If present, record the CreationParameters attribute.
                    try
                        aID = H5A.open(obj.GIDS8, 'CreationParameters');
                        obj.CreationParameters = transpose(H5A.read(aID));
                        H5A.close(aID)
                        
                    catch
                        
                    end % try
                end % if
            end % if
        end % SurpassObjectReader

        function delete(obj)
            % Delete Destructor function for SurpassObjectReader class
            % 
            %
            
            %% Close the HDF5 group.
            H5G.close(obj.GIDS8)
            H5G.close(obj.GID)
        end % delete
        
        function stats = GetStatistics(obj, varargin)
            % GetStatistics Get the statistics for the object
            %
            %   stats = obj.GetStatistics returns the statistics for the
            %   SurpassObjectReader object. The output stats is a struct
            %   with fields for each statistic name, values, and the IDs of
            %   the associated objects.
            %
            %   stats = obj.GetStatistics('List', {str1, str2, str3})
            %   returns the statistics indicated by strings str1, str2,
            %   str3. The strings must match the name of an Imaris
            %   statistic, but are not case sensitive.
            %
            %   stats = obj.GetStatistics('Type', str) returns statistics
            %   with a type matching str, a case-insensitive string from
            %   the list: 'all', 'id', 'singlets', 'tracks' or 'summary'.
            %
            %   stats = obj.GetStatistics('Units', true) returns the
            %   statistics with a field indicating the measurement units
            %   for each statistical type.
            
            %% Parse the inputs.
            parser = inputParser;
            
            parser.addRequired(...
                'obj', @(arg)isa(arg, 'SurpassObjectReader'))
            
            parser.addParameter(...
                'List', [], @(arg)iscellstr(arg)) 
            
            validStringArgCell = {'all', 'id', 'singlet', 'summary', 'track'};
            validateStatReturnArg = ...
                @(arg)any(strcmpi(arg, validStringArgCell));
            parser.addParameter(...
                'Type', 'All', @(arg)validateStatReturnArg(arg)) 
            
            parser.addParameter(...
                'Units', false, @(arg)isnumeric(arg)) 
            
            parser.parse(obj, varargin{:})
            
            %% If the object doesn't have calculated statistics, return.
            if isempty(obj.GIDS8)
                stats = [];
                
                FID = H5I.get_file_id(obj.GID);
                filename = H5F.get_name(FID);
                H5F.close(FID)
                
                stringNoMatch = sprintf(...
                    'No statistics for ''%s'' in file:\n%s', ...
                    obj.Name, ...
                    filename);
                disp(stringNoMatch)
                
                return
            end % if
                        
            %% Read and organize the factor data.
            % Read the Factor HDF5 data.
            DID = H5D.open(obj.GIDS8, 'Factor');
            dataFactor = H5D.read(DID);
            H5D.close(DID)
                        
            % Get the factor ID.
            factorID = dataFactor.ID_List;
            
            % Get the factor name.
            factorName = dataFactor.Name';
            factorName = num2cell(factorName, 2);
            factorName = deblank(factorName);
            
            % Get the factor level.
            factorLevel = dataFactor.Level';
            factorLevel = num2cell(factorLevel, 2);
            factorLevel = deblank(factorLevel);
            
            %% Read and organize the type data.
            % Read the StatisticsType HDF5 data.
            DID = H5D.open(obj.GIDS8, 'StatisticsType');
            dataStatisticsType = H5D.read(DID);
            H5D.close(DID)
            
            % Organize the stat IDs, Names and factors.
            [typeID, uIdxs] = unique(dataStatisticsType.ID);

            typeName = dataStatisticsType.Name';
            typeName = num2cell(typeName, 2);
            typeName = deblank(typeName);
            typeName = typeName(uIdxs);
            
            typeIDFactorList = dataStatisticsType.ID_FactorList;
            typeIDFactorList = typeIDFactorList(uIdxs);
            
            %% If specific stats were requested, trim the stats to gather.
            if ~isempty(parser.Results.List)
                % Get the list and make sure it is a column vector.
                subStatList = parser.Results.List;
                
                if iscolumn(subStatList)
                    subStatList = transpose(subStatList);
                end % if
                
                % Construct the search string.
                regexpCellString = [...
                    repmat({'^'}, size(subStatList)); ...
                    subStatList; ...
                    repmat({'$'}, size(subStatList)); ...
                    repmat({'|'}, size(subStatList))];
                regexpString = [regexpCellString{:}];
                regexpString = regexpString(1:end - 1);
                
                % Get the matching stat name indices.
                subStatMask = ~cellfun(@isempty, ...
                    regexp(typeName, regexpString, ...
                    'Start', 'Once', ...
                    'ignorecase'));
                
                % If no stats match the the requested names, return.
                if ~any(subStatMask)
                    stats = [];
                    
                    FID = H5I.get_file_id(obj.GID);
                    filename = H5F.get_name(FID);
                    H5F.close(FID)
                    
                    stringNoMatch = sprintf(...
                        'No statistics match requested names in file:\n%s', ...
                        filename);
                    disp(stringNoMatch)
                    
                    return
                end % if
                
                % Mask the name, id and factor level idx variables.
                typeName = typeName(subStatMask);
                typeID = typeID(subStatMask);
                typeIDFactorList = typeIDFactorList(subStatMask);
            end % if
            
            %% Read all the statistic values.
            DID = H5D.open(obj.GIDS8, 'StatisticsValue');
            dataStatisticsValue = H5D.read(DID);
            H5D.close(DID)
            
            %% Group the object IDs.
            groups = mapids(unique(dataStatisticsValue.ID_Object), obj);
            
            %% Get the timepoints for per time point stats.
            idTime = unique(dataStatisticsValue.ID_Time);
            nTimepoints = sum(idTime ~= -1);
            
            %% Sort the stat data.
            stats(1:length(typeID)) = struct(...
                'Name', [], ...
                'ID', [], ..., 
                'Value', []);

            for s = 1:length(typeName)
                % If the stat is channel associated, add the channel
                % index to the stat name.
                factorChannelString = factorLevel(...
                    factorID == typeIDFactorList(s) & strcmp(factorName, 'Channel'));

                if ~isempty(factorChannelString)
                    stats(s).Name = [typeName{s} ' Channel ' ...
                        factorChannelString{:}];

                else
                    stats(s).Name = typeName{s};

                end % if

                % Get the stat values.
                sMask = dataStatisticsValue.ID_StatisticsType == typeID(s);
                sValues = dataStatisticsValue.Value(sMask);

                % Handle 'Number of' and 'per time point' stats
                % differently.
                switch regexp(typeName{s}, ...
                        '(Number of)|(per Time Point$)', ...
                        'Match', 'Once')
                
                    case 'Number of'
                        sIDs = dataStatisticsValue.ID_Object(sMask);
                        stats(s).ID = sIDs;
                        
                    case 'per Time Point'
                        % Assign the time point indices to the IDs.
                        sIDs = -ones(nTimepoints, 1, 'int64');
                        stats(s).ID = sIDs;
                        
                    otherwise
                        % Get the object IDs for the values.
                        sIDs = dataStatisticsValue.ID_Object(sMask);
                        stats(s).ID = groups(reduceids(sIDs(1), obj));

                end % if
                
                % Sort and add the values to the struct.
                [~, sOrder] = sort(sIDs);
                sIDMask = ismember(stats(s).ID, sIDs);
                stats(s).Value = nan(size(stats(s).ID));
                stats(s).Value(sIDMask) = sValues(sOrder);
            end % for s
                
            %% Return stat units if requested.
            if parser.Results.Units
                % Format the units.
                statUnits = dataStatisticsType.Unit';
                statUnits = num2cell(statUnits, 2);
                statUnits = deblank(statUnits);
                
                % Mask if necessary.
                if ~isempty(parser.Results.StatList)
                    statUnits = statUnits(subStatMask);                
                end % if
                
                % Add the units to each struct index.
                [stats(1:length(typeID)).Units] = deal(statUnits{:});
            end % if
            
            %% Return a subset of the statistics if requested.
            switch lower(parser.Results.Type)
                
                case 'id'
                    stats = stats(...
                        arrayfun(@(s)...
                        rem(floor(double(s.ID(1))/1e8), 100) >= 0, ...
                        stats));
                    
                case 'overall'
                    stats = stats(arrayfun(@(s)...
                        s.ID(1) == -1, stats));
                   
                case 'singlet'
                   stats = stats(...
                        arrayfun(@(s)...
                        ismember(rem(floor(double(s.ID(1))/1e8), 100), ...
                        [0, 1, 6, 7, 8]), ...
                        stats));
                    
                case 'track'
                     stats = stats(...
                        arrayfun(@(s)...
                        rem(floor(double(s.ID(1))/1e8), 100) >= 10, ...
                        stats));
                   
            end % switch
            
            %% Order the struct elements alphabetically by field name.
            [~, order] = sort({stats.Name});
            stats = stats(order);
        end % GetStatistics
        
        function edges = GetTrackEdges(obj)
            % GetTrackEdges Get the edges (connections) between objects
            %
            %   edges = obj.GetTrackEdges returns an mx2 array containing
            %   the track edges (connections) for the objects.
            
            %% Open the Edges dataset and read it if present.
            if H5L.exists(obj.GID, 'Edges', 'H5P_DEFAULT')
                % Read the edges.
                DID = H5D.open(obj.GID, 'Edges');
                edges = transpose(H5D.read(DID));
                H5D.close(DID)

            else
                edges = [];

            end % if
        end % GetTrackEdges
                
        function ids = GetTrackIDs(obj)
            % GetTrackIDs Get the track to which each edge belongs
            %
            %   ids = obj.GetTrackIDs returns a vector containing
            %   the track for each edge (connected pair of objects).
            
            %% Check for track data.
            if H5L.exists(obj.GIDS8, 'Track0', 'H5P_DEFAULT')
                %% Read the Track0 dataset.
                DID = H5D.open(obj.GIDS8, 'Track0');
                dataTrack0 = H5D.read(DID);
                H5D.close(DID)

                %% Construct the track ID list.
                trackIDs = arrayfun(...
                    @(c, a, b)repmat(c, [1, b - a]), ...
                    dataTrack0.ID', ...
                    dataTrack0.IndexTrackEdgeBegin', ...
                    dataTrack0.IndexTrackEdgeEnd', ...
                    'UniformOutput', 0);

                ids = transpose([trackIDs{:}]);

            else
                ids = [];
                
            end % if
        end % GetTrackIDs
        
    end % methods
    
    methods (Abstract = true)
        
        GetIndicesT(obj) % Get the time indices of the objects
        
    end % methods (Abstract)
    
    events
        
    end % events
    
end % class SurpassObjectReader


%% Subfunction to compare identifiers
% H5L.iterate chokes if this is placed as a static method. Boo!
function [status, matchId] = matchimarisids(GID, lName, matchId)
    % MATCHIMARISIDS Compare the /Scene object Id to /Scene8 object Ids
    %
    %

    %% Get the Id for the group.
    groupID = H5G.open(GID, lName);
    aID = H5A.open(groupID, 'Id');
    statId = H5A.read(aID);

    %% Close the HDF5 objects.
    H5A.close(aID)
    H5G.close(groupID)

    %% If we've found the match, set the status to 1 to end iteration.
    if matchId == statId
        status = 1;

    else
        status = 0;

    end % if
end % matchimarisids
