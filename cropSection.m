


dir_input = './output/';%'/om/user/dgoodwin/ExSeq/';
dir_rootoutput = './output/';
experiment_string = 'sa0916dncv';
PREPEND = 'CROP';
TOP_LEFT = [1125,291];
BOTTOM_RIGHT = [1699,687];

if ~exist(dir_rootoutput, 'dir')
    mkdir(dir_rootoutput);
end

files = dir(dir_input);
files = files([files.isdir]); %only load the output

files(1:2) = []; %ignore . and ..

%%

for file_index = 1:length(files)
    %Don't recrop the same fil
    if ~findstr(files(file_index).name,'TPS') %Make sure it's an image output
        continue;
    end
   
    %load
    data = loadTifSequence(fullfile(dir_input,files(file_index).name));
    %crop
    data = data(TOP_LEFT(1):BOTTOM_RIGHT(1),TOP_LEFT(2):BOTTOM_RIGHT(2),:);
    %save
    save3DTif(data,fullfile(dir_rootoutput,[PREPEND files(file_index).name '.tif']));
end
