classdef SpotsReader < SurpassObjectReader
    % SpotsReader Read Imaris ims file Spots data
    %
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader | SurpassObjectReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        NumberOfSpots % Number of spots in the object
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    methods
        
        function obj = SpotsReader(GID)
            % SpotsReader Create a Spots reader object
            %
            %   obj = SpotsReader(GID) constructs obj to read Imaris Spots
            %   data. GID is an HDF5 Group Identifier for a Spots (/Points)
            %   group in an ims file.
            
            %% Call the superclass constructor.
            obj = obj@SurpassObjectReader(GID, 'Spots');
            
            %% Check for the required data sets for a valid Spots object.
            isValidSpots = ...
                H5L.exists(GID, 'CoordsXYZR', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'Time', 'H5P_DEFAULT') && ...
                H5L.exists(GID, 'TimeInfos', 'H5P_DEFAULT');
            
            %% If the group is a valid Spots group, determine the number of spots.
            if isValidSpots
                % Determine the number of spots.
                DID = H5D.open(GID, 'Time');
                FSID = H5D.get_space(DID);
                [~, spaceDims, ~] = H5S.get_simple_extent_dims(FSID);
                obj.NumberOfSpots = spaceDims(1);
            
                % Close the HDF5 dataset objects.
                H5S.close(FSID);
                H5D.close(DID)
            end % if
        end % SpotsReader
        
        function ids = GetIDs(obj)
            % GetIDs Get the IDs for all spots
            %
            %   ids = obj.GetIDs returns a vector containing the IDs of all
            %   the spots in the Spots object.
            
            %% Read the ID field from the Surface dataset.
            DID = H5D.open(obj.GIDS8, 'Spot');
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
            % GetIndicesT Get the time indices of the spots
            %
            %   idx = obj.GetIndicesT returns a vector of zero-based
            %   indices indicating the time index of each spot.

            %% Open the Time dataset.
            DID = H5D.open(obj.GID, 'Time');
            idx = transpose(H5D.read(DID));
            
            %% Close the HDF5 dataset object.
            H5D.close(DID)
        end % GetIndicesT
        
        function pos = GetPositions(obj)
            % GetPositions Get the xyz positions of the spots
            %
            %   pos = obj.GetPositions returns an mx3 array of spot
            %   centroids.
            
            %% Open the Coordinates dataset.
            DID = H5D.open(obj.GID, 'CoordsXYZR');
            dataCoordsXYZR = transpose(H5D.read(DID));
            pos = dataCoordsXYZR(:, 1:3);
            
            %% Close the HDF5 dataset object.
            H5D.close(DID)
        end % GetPositions

        function radii = GetRadii(obj)
            % GetRadii Get the radii of the spots
            %
            %   radii = obj.GetRadii returns a vector of spot radii.
            
            %% Check for ellipsoid spots.
            % If there is a RadiusYZ dataset, the spots are ellipsoids.
            % Calculate the radius as the weighted average of the lateral
            % and axial radii.
            if H5L.exists(obj.GID, 'RadiusYZ', 'H5P_DEFAULT')
                %% Read the YZ radii.
                DID = H5D.open(obj.GID, 'RadiusYZ');
                dataRadiusYZ = H5D.read(DID);
                H5D.close(DID);
                
                %% Calculate the weighted average radii.
                radii = transpose(...
                    arrayfun(...
                    @(y, z)mean([y, y, z]), ...
                    dataRadiusYZ(1, :), ...
                    dataRadiusYZ(2, :)));
                
            else
                %% Read the Coordinates dataset and return the 4th row.
                DID = H5D.open(obj.GID, 'CoordsXYZR');
                dataCoordsXYZR = H5D.read(DID);
                H5D.close(DID)
                
                radii = transpose(dataCoordsXYZR(4, :));
                
            end % if
        end % GetRadii
        
        function radii = GetRadiiXYZ(obj)
            % GetRadiiXYZ Get the xyz radii of the spots
            %
            %   radiiXYZ = obj.GetRadii returns an mx3 array of spot
            %   xyz radii.
            
            %% Read the YZ radii if present.
            if H5L.exists(obj.GID, 'RadiusYZ', 'H5P_DEFAULT')
                DID = H5D.open(obj.GID, 'RadiusYZ');
                dataRadiusYZ = transpose(H5D.read(DID));
                H5D.close(DID);
                
                radii = [dataRadiusYZ(:, 1), dataRadiusYZ];
                
            else
                %% Read the Coordinates dataset and return the 4th row.
                DID = H5D.open(obj.GID, 'CoordsXYZR');
                dataCoordsXYZR = H5D.read(DID);
                H5D.close(DID)
                
                radii = repmat(...
                    transpose(dataCoordsXYZR(4, :)), ...
                    [1, 3]);
                                
            end % if 
        end % GetRadiiXYZ

    end % methods

    events 
        
    end % events
    
end % class SpotsReader