function [row, col] = getTilePositionFromTerastitcherXML(tileNum, xmlStruct)
    %getTilePositionFromTerastitcherXML: Process terastitcher input xml
    %file, which includes the row and column orders from. The input,
    %xmlStruct was loaded by parseXML.m, which was originally taken from
    %the MATLAB documentation for reading XML files
    
    %Loop over high-level XML document to find the STACKS child 
    
    stacks_idx = 0;
    for child_idx = 1:length(xmlStruct.Children)
        if strcmp(xmlStruct.Children(child_idx).Name,'STACKS')
           stacks_idx = child_idx;
           continue
        end
    end
    
    if ~stacks_idx
        error('Unable to process XML file properly');
    end
    
    %Now loop through all the children of the STACKS section to find the
    %one that contains the field of view we're looking for, assuming the
    %F.3i format
    tileString = sprintf('F%.3i',tileNum);
    for i = 1:length(xmlStruct.Children(stacks_idx).Children)
        if strcmp(xmlStruct.Children(stacks_idx).Children(i).Name,'Stack')
            %Step over all the attributes looking for the filename, which
            %is stored in the IMG_REGEX atttribue
            for j = 1:length(xmlStruct.Children(stacks_idx).Children(i).Attributes)
                if strcmp(xmlStruct.Children(stacks_idx).Children(i).Attributes(j).Name,...
                        'IMG_REGEX')
                   filename = xmlStruct.Children(stacks_idx).Children(i).Attributes(j).Value;
                   
                   %If we've found the entry for the file, now get the row
                   %and column!
                   if contains(filename,tileString)
                        
                       row_idx = findAttributeIndex(xmlStruct.Children(stacks_idx).Children(i),'ROW');
                       col_idx = findAttributeIndex(xmlStruct.Children(stacks_idx).Children(i),'COL');
                       row = str2num(xmlStruct.Children(stacks_idx).Children(i).Attributes(row_idx).Value);
                       col = str2num(xmlStruct.Children(stacks_idx).Children(i).Attributes(col_idx).Value);
                       return
                   end
                   
                end %End if statement if we found the 
            end % End loop over a stacks attributes
        end %End if logic to check if it is a stack
    end %End loop over all the stacks children

    row = -1;
    col = -1;
end

function idx = findAttributeIndex(xmlChild, attName)
    idx = -1;
    for j = 1:length(xmlChild.Attributes)
        if strcmp(xmlChild.Attributes(j).Name,attName)
            idx = j;
        end
    end
    
end
