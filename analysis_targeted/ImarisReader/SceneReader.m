classdef SceneReader < matlab.mixin.SetGet & dynamicprops
    % DatasetReader Read Imaris ims file Surpass Scene information
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)

        GID % HDF5 file Group ID for the Dataset group
        NumberOfChildren % Total number of objects in the Surpass Scene
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    properties (SetAccess = 'private', GetAccess = 'public', Transient = true)
        
        NumberOfCells % Number of Cell objects in the Surpass Scene
        NumberOfFilaments % Number of Filaments objects in the Surpass Scene
        NumberOfSpots % Number of Spots objects in the Surpass Scene
        NumberOfSurfaces % Number of Surfaces objects in the Surpass Scene
        
    end % properties (SetAccess = 'private', GetAccess = 'public', Transient = true)
    
    methods
            
        function obj = SceneReader(GID)
            % SceneReader Create a scene reader object
            %
            % obj = SceneReader(GID) constructs obj to read Imaris Surpass
            % Scene properties. GID is an HDF5 Group Identifier for the
            % /Scene group in an ims file.
            
            %% Store the dataset groupd ID.
            obj.GID = GID;
            
            %% Determine the number of children.
            obj.NumberOfChildren = H5G.get_num_objs(obj.GID);
            
            %% Read the attributes to get the number of objects.
            [~, ~, obj] = H5A.iterate(obj.GID, ...
                'H5_INDEX_NAME', ...
                'H5_ITER_NATIVE', ...
                0, ...
                @obj.readattributes, ...
                obj);
        end % SceneReader
        
        function delete(obj)
            % Delete Destructor function for SceneReader
            %
            %
            
            %% Close the HDF5 GID.
            H5G.close(obj.GID)            
        end % delete
        
    end % methods
    
    methods (Static, Hidden)
        
        function [status, obj] = readattributes(GID, aName, ~, obj)
            % readattributes Read the attributes for each channel 

            %% Open the attribute by name.
            aID = H5A.open_by_name(GID, '/Scene/Content', aName);

            %% Read the attribute value.
            switch aName

                case 'NumberOfCells'
                    obj.NumberOfCells = H5A.read(aID);

                case 'NumberOfFilaments'
                    obj.NumberOfFilaments = H5A.read(aID);

                case 'NumberOfPoints'
                    obj.NumberOfSpots = H5A.read(aID);

                case 'NumberOfSurfaces'
                    obj.NumberOfSurfaces = H5A.read(aID);

            end % switch

            H5A.close(aID);

            %% Set the status to continue.
            % A value of 0 for status tells H5A.iterate to continue or
            % return if all attributes processed.
            status = 0;
        end % readattributes
        
    end % methods (Access = 'private')
    
    events
        
    end % events
    
end % class SceneReader
