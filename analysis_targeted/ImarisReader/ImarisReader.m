classdef ImarisReader < matlab.mixin.SetGet & dynamicprops
    % ImarisReader Imaris ims file reader class
    %
    %   Syntax
    %   ------
    %   obj = ImarisReader(filepath)
    %
    %   Description
    %   -----------
    %   obj = ImarisReader(filepath) creates an ImarisReader object obj to
    %   read the image and scene data in the file represented by the string
    %   filepath.
    %
    %   ImarisReader provides access to the data stored in an Imaris ims
    %   file. ImarisReader can be used to access the image data, as well as
    %   the Surpass Scene object data.
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also DatasetReader | CellsReader | SpotsReader | SurfacesReader | FilamentsReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        DataSet % DatasetReader object to access the Imaris DataSet
        FID % HDF5 File identifier
        Scene % SceneReader object to access the Surpass Scene properties
        Cells % CellReader objects to access Filament object properties
        Filaments % FilamentsReader objects to access Filament object properties
        Spots % SpotsReader objects to access Spots object properties
        Surfaces % SurfacesReader objects to access Surfaces object properties
        
    end % properties
    
    methods
        
        function obj = ImarisReader(imsFile)
            % IMARISREADER Constructor function
            %
            % imsObj = imsreader(filename) constructs the ImarisReader
            % object imsObj to read data from the file represented by
            % filename. The input filename is a single quotation mark
            % enclosed string that specifies an ims file on the MATLAB
            % path.
            
            %% Check that the input is a valid file.
            parser = inputParser;
            parser.addRequired('imsFile', ...
                @(arg)exist(arg, 'file') && strcmp(arg(end - 2:end), 'ims'))
            
            parse(parser, imsFile);
            
            %% Open the file.
            obj.FID = H5F.open(imsFile, ...
                'H5F_ACC_RDONLY', ...
                'H5P_DEFAULT');
            
            %% Get the groups in the file.
            nGroups = H5G.get_num_objs(obj.FID);
            groupNames = cell(nGroups, 1);
            for g = 1:nGroups
                groupNames{g} = H5L.get_name_by_idx(obj.FID, '/', ...
                    'H5_INDEX_NAME', ...
                    'H5_ITER_INC', ...
                    g - 1, ...
                    'H5P_DEFAULT');
            end % for g
            
            %% Create the dataset reader object.
            obj.DataSet = DatasetReader(H5G.open(obj.FID, '/DataSet'));
            
            %% If there is no Surpass scene in the file, return.
            if all(cellfun(@isempty, regexp(groupNames, 'Scene', 'Start', 'Once')))
                return
            end % if
            
            %% Get the Surpass objects.
            obj.Scene = SceneReader(H5G.open(obj.FID, '/Scene/Content'));
            
            for c = 1:obj.Scene.NumberOfChildren
                % Open the nth group.
                cGID = H5O.open_by_idx(obj.FID, '/Scene/Content', ...
                    'H5_INDEX_NAME', ...
                    'H5_ITER_INC', ...
                    c - 1, ...
                    'H5P_DEFAULT');
                
                % Get the HDF5 name.
                hdf5Name = H5L.get_name_by_idx(obj.FID, '/Scene/Content', ...
                    'H5_INDEX_NAME', ...
                    'H5_ITER_INC', ...
                    c - 1, ...
                    'H5P_DEFAULT');
                
                strIdentifier = regexp(hdf5Name, ...
                    'Cells|Filaments|Points|Surfaces', 'Match', 'Once');
                
                % Create the appropriate Surpass reader object.
                switch strIdentifier
                
                    case 'Cells'
                        cCells = CellsReader(cGID);
                    
                        % If the Cells group is incomplete, don't add it.
                        if isempty(cCells.NumberOfCells)
                            continue
                        end % if
                        
                        if isempty(obj.Cells)
                            obj.Cells = cCells;
                            
                        else
                            obj.Cells(end + 1) = cCells;
                        
                        end % if
                        
                    case 'Filaments'
                        % Read the filaments group.
                        cFilaments = FilamentsReader(cGID);
                        
                        % If the Filaments group is incomplete, don't add
                        % it.
                        if isempty(cFilaments.NumberOfFilaments)
                            continue
                        end % if
                        
                        % Add it to the ImarisReader object.
                        if isempty(obj.Filaments)
                            obj.Filaments = cFilaments;
                            
                        else
                            obj.Filaments(end + 1) = cFilaments;
                        
                        end % if
                        
                    case 'Points'
                        % Read the data in the Spots group.
                        cSpots = SpotsReader(cGID);
                        
                        % If the Spots group is incomplete, don't add it.
                        if isempty(cSpots.NumberOfSpots)
                            continue
                        end % if
                        
                        if isempty(obj.Spots)
                            obj.Spots = cSpots;
                            
                        else
                            obj.Spots(end + 1) = cSpots;
                        
                        end % if
                        
                    case 'Surfaces'
                        % Read the Surfaces group.
                        cSurfaces = SurfacesReader(cGID);
                        
                        % If the Surfaces group is incomplete, don't add
                        % it.
                        if isempty(cSurfaces.NumberOfSurfaces)
                            continue
                        end % if
                        
                        if isempty(obj.Surfaces)
                            obj.Surfaces = cSurfaces;
                            
                        else
                            obj.Surfaces(end + 1) = cSurfaces;
                        
                        end % if
                        
                end % switch strIdentifier
            end % for n
        end % ImarisReader 
        
        function delete(obj)
            % DELETE Destructor function or ImarisReader
            %
            %
            
            %% Close the HDF5 file access object.
            H5F.close(obj.FID)
        end % delete
        
    end % methods
    
    events
        
    end % events
    
end % class ImarisReader