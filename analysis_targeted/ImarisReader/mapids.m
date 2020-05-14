function mapped = mapids(ids, obj)
    % mapids Map Imaris IDs to object types
    %
    %   Syntax
    %   ------
    %   mapped = mapids(ids, obj)
    %
    %   Description
    %   -----------
    %   mapped = groupids(ids, obj) creates a containers.Map object that
    %   assigns object IDs to keys that correspond to the type of object as
    %   determined by the object IDs. The input ids is a list of Imaris IDs
    %   for objects such as Spots, Tracks, etc. The input obj is a
    %   SurpassObjectReader object. The returned container maps IDs to keys
    %   that represent the object type as calculated by the reduceids
    %   function.
    %
    %   © 2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also reduceids | ImarisReader | SurpassObjectReader | containers.Map

    %% Parse the inputs.
    parser = inputParser;
    parser.addRequired('ids', @(arg)isvector(arg))
    parser.addRequired('obj', @(obj)(isa(obj, 'SurpassObjectReader')))
    parser.parse(ids, obj)
    
    %% Reduce the IDs to numerical ID classes.
    reduced = reduceids(ids, obj);
    
    %% Create the map.
    switch class(obj)
        
        case 'CellsReader'
            keySet = -1:19;
            
        case 'SpotsReader'
            keySet = [-1, 0, 10];
        
        case 'SurfacesReader'
            keySet = [-1, 0, 10];
        
        case 'FilamentsReader'
            keySet = [-1, 1, 2, 6:8, 11, 18];
            
    end % switch
    
    %% Organize the IDs by their reduced ID.
    valueSet = cell(size(keySet));
    for k = 1:numel(keySet)
        valueSet{k} = ids(reduced == keySet(k));
    end % for f

    mapped = containers.Map(keySet, valueSet);
end % bunchids