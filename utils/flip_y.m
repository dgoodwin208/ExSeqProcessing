
INPUT_DIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampusOrig/1_original/'

files = dir(fullfile(INPUT_DIRECTORY,'*.tif'));

fprintf('Source file\tDestination file\n')
for file_indx = 1:length(files)
    files(file_indx).name
    img = load3DTif_uint16(fullfile(INPUT_DIRECTORY,files(file_indx).name));
    img = img(end:-1:1,:,:);
    save3DTif_uint16(img,fullfile(INPUT_DIRECTORY,files(file_indx).name));
end
