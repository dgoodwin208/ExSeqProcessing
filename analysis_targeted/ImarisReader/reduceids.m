function reduced = reduceids(ids, obj)
    % reduiceids Reduce Imaris IDs to a fundamental numerical identifier
    %
    %   Syntax
    %   ------
    %   reduced = reduceids(ids)
    %
    %   Description
    %   -----------
    %   reduced = classifyids(ids, obj) converts the Imaris object IDs in
    %   ids to the numerical values that represent all data objects
    %   matching the object type for the SurpassObjectReader obj.
    %
    %   Notes
    %   -----
    %   The object type can be deduced from Imaris IDs using the table:
    %
    %   Object type                 Minimum ID	Maximum ID	Reduced ID
    %   -----------------------     ----------  ----------  ---------- 
    %   Overall                             -1          -1          -1
    %   Spot/Surface/Cell                    0     1e8 - 1           0
    %   Filament/Nucleus                   1e8     2e8 - 1           1
    %   Vesicle                            2e8    10e8 - 1         2?9
    %   Dendrite*                        x06e8   x07e8 - 1           6
    %   Spine*                           x07e8   x08e8 - 1           7
    %   Point                              8e8     9e8 - 1           8
    %   Spot/Surface/Cell Track           10e8    11e8 - 1          10
    %   Filament/Nucleus Track            11e8    12e8 - 1          11
    %   Vesicle Track                     12e8    19e8 - 1       12?19
    %   Point Track                       18e8    19e8 - 1          18
    %   
    %   *Dendrite and Spine IDs are prepended with a number x that
    %    indicates the parent filament for the Dendrite or Spine.
    %
    %   © 2016, Peter Beemiller (pbeemiller@gmail.com)
    %
    %   See also ImarisReader | SurpassObjectReader

    %% Parse the inputs.
    parser = inputParser;
    parser.addRequired('ids', @(arg)isvector(arg))
    parser.addRequired('obj', @(obj)(isa(obj, 'SurpassObjectReader')))
    parser.parse(ids, obj)
    
    %% Divide by 1e8, then take the remainder after division by 100.
    reduced = rem(floor(double(ids)/1e8), 100);
end % reduceids