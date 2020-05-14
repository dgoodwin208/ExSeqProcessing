classdef DatasetReader < matlab.mixin.SetGet & dynamicprops
    % DatasetReader Read Imaris ims file DataSets
    % 
    %   © 2015–2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader
    
    properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
        
        GID % HDF5 file Group ID for the Dataset group
        DataType % Numeric data type for the data set 
        ExtendMinX % Minimum extent of the dataset in the x dimension
        ExtendMinY % Minimum extent of the dataset in the y dimension
        ExtendMinZ % Minimum extent of the dataset in the z dimension
        ExtendMaxX % Maximum extent of the dataset in the x dimension
        ExtendMaxY % Maximum extent of the dataset in the y dimension
        ExtendMaxZ % Maximum extent of the dataset in the z dimension
        SizeX % Number of voxels in the x dimension
        SizeY % Number of voxels in the y dimension
        SizeZ % Number of voxels in the z dimension
        SizeC % Number of channels in the dataset
        SizeT % Number of time points in the dataset
        ChannelInfo = struct() % Channel attributes for the dataset, including Color, Display Range, Name, etc.
        Timestamps = cell(1, 1) % Timestamps strings for the time points stored as 'YYYY-MM-DD HH:MM:SS.SSS'
        
    end % properties (SetAccess = 'immutable', GetAccess = 'public', Transient = true)
    
    methods
        
        function obj = DatasetReader(GID)
            % DatasetReader Create a dataset reader object
            %
            %   obj = DatasetReader(GID) constructs obj to read Imaris
            %   DataSet image data. GID is an HDF5 Group Identifier for the
            %   /DataSet group in an ims file.
            
            %% Store the dataset groupd ID.
            obj.GID = GID;
            
            %% Get the DataSet Info sub-groups.
            % The Imaris DataSet is named 'DataSet'. Cell Mask DataSets will
            % be named 'DataSet#', where # is the DataSet Id.
            stringDataSetNumber = regexp(H5I.get_name(GID), ...
                '\d{1,}$', 'Match', 'Once');
            
            FID = H5I.get_file_id(GID);
            infoGID = H5G.open(FID, ['/DataSetInfo' stringDataSetNumber]);
            H5F.close(FID)
            
            % Replace this with an H5L.iterate call looking for groups with
            % channel in the name.
            nGroups = H5G.get_num_objs(infoGID);
            groupNames = cell(nGroups, 1);
            for g = 1:nGroups
                groupNames{g} = H5L.get_name_by_idx(infoGID, ...
                    ['/DataSetInfo' stringDataSetNumber], ...
                    'H5_INDEX_NAME', ...
                    'H5_ITER_NATIVE', ...
                    g - 1, ...
                    'H5P_DEFAULT');
            end % for g
            
            %% Determine the number of channels from the group names.
            obj.SizeC = sum(~cellfun(@isempty, ...
                regexp(groupNames, 'Channel \d{1,}', 'Start', 'Once')));

            %% Create a list of all attributes for the channels.
            for c = 1:obj.SizeC
                cGID = H5G.open(infoGID, ['Channel ' num2str(c - 1)]);
                [~, ~, obj.ChannelInfo] = H5A.iterate(cGID, ...
                    'H5_INDEX_NAME', ...
                    'H5_ITER_NATIVE', ...
                    0, ...
                    @obj.findchannelattributes, ...
                    obj.ChannelInfo);
                H5G.close(cGID)
            end % for c
            
            %% Get the field names and use them to create a blank struct.
            fieldNames = fieldnames(obj.ChannelInfo);
            defaultValues = repmat({[]}, size(fieldNames));
            structCreationCell = [fieldNames, defaultValues]';
            cInfo = struct(structCreationCell{:});

            %% Read the properties for each channel.
            for c = 1:obj.SizeC
                cCell = {cInfo, c - 1};
                cGID = H5G.open(infoGID, ['Channel ' num2str(c - 1)]);
                [~, ~, cCell] = H5A.iterate(cGID, ...
                    'H5_INDEX_CRT_ORDER', ...
                    'H5_ITER_INC', ...
                    0, ...
                    @obj.readchannelattributes, ...
                    cCell);
                H5G.close(cGID)
                obj.ChannelInfo(c) = cCell{1};
            end % for c
            
            %% Get the image xyz dimensions and spatial extents.
            imageGID = H5G.open(infoGID, 'Image');
            
            % Read the xyz dimensions.
            aID = H5A.open(imageGID, 'X');
            obj.SizeX = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'Y');
            obj.SizeY = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'Z');
            obj.SizeZ = str2double(H5A.read(aID));
            H5A.close(aID)
            
            % Read in the extent minimums and maximums.
            aID = H5A.open(imageGID, 'ExtMin0');
            obj.ExtendMinX = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMax0');
            obj.ExtendMaxX = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMin1');
            obj.ExtendMinY = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMax1');
            obj.ExtendMaxY = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMin2');
            obj.ExtendMinZ = str2double(H5A.read(aID));
            H5A.close(aID)
            
            aID = H5A.open(imageGID, 'ExtMax2');
            obj.ExtendMaxZ = str2double(H5A.read(aID));
            H5A.close(aID)
            
            H5G.close(imageGID);
            
            %% Get the number of time points and read the time stamps.
            level0GID = H5G.open(obj.GID, 'ResolutionLevel 0');
            gInfo = H5G.get_info(level0GID);
            H5G.close(level0GID)
            
            obj.SizeT = gInfo.nlinks;
            obj.Timestamps = cell(obj.SizeT, 1);
            
            tInfoGID = H5G.open(infoGID, 'TimeInfo');
            for t = 1:obj.SizeT
                tAID = H5A.open(tInfoGID, ['TimePoint' num2str(t)]);
                obj.Timestamps{t} = H5A.read(tAID);
                close(tAID)
            end % for t
            
            H5G.close(tInfoGID)
            H5G.close(infoGID)
            
            %% Get the dataset data type.
            dataGID = H5G.open(obj.GID, 'ResolutionLevel 0/TimePoint 0/Channel 0');
            dataDID = H5D.open(dataGID, 'Data');
            dataTID = H5D.get_type(dataDID);
            
            if H5T.equal(dataTID, 'H5T_STD_U8LE')
                obj.DataType = 'uint8';
                
            elseif H5T.equal(dataTID, 'H5T_STD_U16LE')
                obj.DataType = 'uint16';
                
            else
                obj.DataType = 'single';
                
            end % if
            
            H5T.close(dataTID)
            H5D.close(dataDID)
            H5G.close(dataGID)
        end % DatasetReader
        
        function delete(obj)
            % Delete Destructor function for DatasetReader
            %
            %
            
            %% Close the HDF5 GID.
            H5G.close(obj.GID)
        end % delete
        
        function data = GetData(obj)
            % GetData Returns the entire dataset as a 5D array
            %
            %   data = obj.GetData returns a 5D array containing all the
            %   image data in the dataset. The array dimensions correspond
            %   to the xyzct dimensions of the dataset.

            %% Allocate the volume.
            switch obj.DataType
                
                case 'eTypeUInt8'
                    data = zeros(...
                        obj.SizeX, obj.SizeY, obj.SizeZ, obj.SizeC, obj.SizeT, 'uint8');
                    
                case 'eTypeUInt16'
                    data = zeros(...
                        obj.SizeX, obj.SizeY, obj.SizeZ, obj.SizeC, obj.SizeT, 'uint16');
                    
                otherwise
                    data = zeros(...
                        obj.SizeX, obj.SizeY, obj.SizeZ, obj.SizeC, obj.SizeT, 'single');
                    
            end % switch
            
            %% Read the data in all the channel groups inside all the time point groups.
            for t = 1:obj.SizeT
                for c = 1:obj.SizeC
                    %% Construct the path to the dataset location.
                    groupLocation = ['ResolutionLevel 0/TimePoint ' num2str(t - 1)  ...
                        '/Channel ' num2str(c - 1)];
                    groupID = H5G.open(obj.GID, groupLocation);
                    DID = H5D.open(groupID, 'Data');

                    %% Construct the hyper slab to read.
                    slab = [obj.SizeY, obj.SizeX];
                    offset = [0, 0, 0];
                    block = [1, obj.SizeY, obj.SizeX];
                    MSID = H5S.create_simple(2, slab, []);
                    FSID = H5D.get_space(DID);
                    H5S.select_hyperslab(FSID,'H5S_SELECT_SET', offset, [], [], block);            

                    %% Read the data slices into the array.
                    for z = 1:obj.SizeZ
                        data(:, :, z, c, t) = H5D.read(DID,'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT');
                        H5S.select_hyperslab(FSID,'H5S_SELECT_SET', [z 0 0], [], [], block); 
                        % H5S.offset_simple(fileSpaceID, [1 0 0])
                    end % for z
            
                    %% Close the HDF5 objects.
                    H5S.close(MSID)
                    H5S.close(FSID)
                    H5D.close(DID)
                    H5G.close(groupID)
                end % for c
            end % for t
        end % GetData

        function slice = GetDataSlice(obj, zIdx, cIdx, tIdx)
            % GetDataSlice Returns a single image slice from the dataset
            %
            %   dataSlice = obj.GetDataSlice(zIdx, vIdx, tIdx) returns the
            %   data slice for the z position, channel and time point
            %   represented by the zero-based indices zIdx, cIdx and tIdx,
            %   respectively.

            %% Construct the path to the dataset location.
            groupLocation = ['ResolutionLevel 0/TimePoint ' num2str(tIdx)  ...
                '/Channel ' num2str(cIdx)];
            groupID = H5G.open(obj.GID, groupLocation);
            DID = H5D.open(groupID, 'Data');
            
            %% Construct the hyper slab to read.
            slab = [obj.SizeY, obj.SizeX];
            offset = [zIdx, 0, 0];
            block = [1, obj.SizeY, obj.SizeX];
            MSID = H5S.create_simple(2, slab, []);
            FSID = H5D.get_space(DID);
            H5S.select_hyperslab(FSID,'H5S_SELECT_SET', offset, [], [], block);            

            %% Read the data slice from the volume.
            slice = H5D.read(DID,'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT');
            
            %% Close the HDF5 objects.
            H5S.close(MSID)
            H5S.close(FSID)
            H5D.close(DID)
            H5G.close(groupID)
        end % GetDataSlice
        
        function volume = GetDataVolume(obj, cIdx, tIdx)
            % GetDataVolume Returns a single image volume from the dataset
            %
            % dataVolume = obj.GetDataVolume(cIdx, tIdx) returns the
            % data volume (xyz image) for the channel and time point
            % represented by the zero-based indices cIdx and tIdx,
            % respectively.

            %% Allocate the volume.
            switch obj.DataType
                
                case 'eTypeUInt8'
                    volume = zeros(obj.SizeX, obj.SizeY, obj.SizeZ, 'uint8');
                    
                case 'eTypeUInt16'
                    volume = zeros(obj.SizeX, obj.SizeY, obj.SizeZ, 'uint16');
                    
                otherwise
                    volume = zeros(obj.SizeX, obj.SizeY, obj.SizeZ, 'single');
                    
            end % switch
            
            %% Construct the path to the dataset location.
            groupLocation = ['ResolutionLevel 0/TimePoint ' num2str(tIdx)  ...
                '/Channel ' num2str(cIdx)];
            groupID = H5G.open(obj.GID, groupLocation);
            DID = H5D.open(groupID, 'Data');
            
            %% Construct the hyper slab to read.
            slab = [obj.SizeY obj.SizeX];
            offset = [0 0 0];
            block = [1 obj.SizeY obj.SizeX];
            MSID = H5S.create_simple(2, slab, []);
            FSID = H5D.get_space(DID);
            H5S.select_hyperslab(FSID, 'H5S_SELECT_SET', offset, [], [], block);            
            
            %% Read the data slices into the array.
            for z = 1:obj.SizeZ
                % Read the selected hyperslab.
                volume(:, :, z) = ...
                    H5D.read(DID,'H5ML_DEFAULT', MSID, FSID, 'H5P_DEFAULT');
                
                % Select the next hyperslab.
                H5S.select_hyperslab(...
                    FSID,'H5S_SELECT_SET', [z 0 0], [], [], block); 
            end % for z
            
            %% Close the HDF5 objects.
            H5S.close(MSID)
            H5S.close(FSID)
            H5D.close(DID)
            H5G.close(groupID)
        end % GetDataVolume

    end % methods
    
    methods (Access = 'protected', Static = true, Hidden = true)
        
        function [status, ChannelInfo] = findchannelattributes(~, aName, ~, ChannelInfo)
            % findchannelattributes Create a list of all channel attributes
            %
            %

            %% Create a field for the name if needed.
            if ~isfield(ChannelInfo, aName)
                ChannelInfo.(aName) = [];
            end % if
            
            %% Set the status to continue.
            % A value of 0 for status tells H5A.iterate to continue or
            % return if all attributes processed.
            status = 0;
        end % findchannelattributes

        function [status, cCell] = readchannelattributes(GID, aName, ~, cCell)
            % readchannelattributes Reads the attributes for each channel 

            %% Extract the struct and channel index from the input cell.
            cStruct = cCell{1};
            cIdx = cCell{2};

            %% Open the attribute by name.
            aID = H5A.open_by_name(GID, ...
                ['/DataSetInfo/Channel ' num2str(cIdx)], aName);

            %% Read the attribute value.
            switch aName

                case {'Color', 'ColorRange'}
                    cStruct.(aName) = str2num(H5A.read(aID));

                case {'ColorOpacity', 'GammaCorrection'}
                    cStruct.(aName) = str2double(H5A.read(aID));

                otherwise
                    cStruct.(aName) = H5A.read(aID);

            end % switch

            H5A.close(aID);

            %% Repack the struct and channel index for the next iteration.
            cCell = {cStruct, cIdx};

            %% Set the status to continue.
            % A value of 0 for status tells H5A.iterate to continue or
            % return if all attributes processed.
            status = 0;
        end % readchannelattributes
        
    end % methods (Static)
        
    events
        
    end % events
    
end % class DatasetReader