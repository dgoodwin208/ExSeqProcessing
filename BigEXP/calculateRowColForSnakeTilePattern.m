function [row,col] = calculateRowColForSnakeTilePattern(tileNum, numRows)
%CALCULATEROWCOLFORSNAKETILEPATTERN Convert a tile number to a (row,col)
%spatial position using a snaking pattern that goes column wise, starting
%down from top left corner (0,0). Note that tileNum is 0-indexed and that 
%the number of columns is unnecessary

col = floor(tileNum/numRows);

%For the even columns, the number counts upward
if mod(col,2)==0
    row = mod(tileNum,numRows);
else
    row = numRows-mod(tileNum,numRows)-1;
end

end

