function editParams_batch(basename)

% Read txt into cell A
fid = fopen('loadParameters.m','r');
i = 1;
tline = fgetl(fid);
A{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    A{i} = tline;
end
fclose(fid);
% Change first line of loadParameters.m file
A{1} = ['params.FILE_BASENAME = ''' basename ''';'];
% Write cell A into txt
fid = fopen('loadParameters.m', 'w');
for i = 1:numel(A)
    if A{i+1} == -1
        fprintf(fid,'%s', A{i});
        break
    else
        fprintf(fid,'%s\n', A{i});
    end
end
fclose('all');
end

