function crop_and_move_files(round_num)
%interpolateDataVolumeSummary Load data and upsample the data

FILEROOT_NAME_INPUT = 'exseqautoframe7i';
FILEROOT_NAME_OUTPUT = 'exseqautoframe7icrop';

%Taken from Fiji so the XY might be measured from the wrong origin
YRANGE = 451:950;
XRANGE = 1001:1500;
ZRANGE = round(43*(.5/.165)):round(76*(.5/.165)); %upsample for interrpolated 

fprintf('Starting processing of round %i\n',round_num);
%Load all channels, normalize them, calculate the cross corr of
%channels 1-3 vs 4

INPUTDIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampusOrig/4_registration';
OUTPUTDIRECTORY  = '/mp/nas0/ExSeq/AutoSeqHippocampusOrig/4_registration-cropped';

chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

for chan_idx = 1:length(chan_strs)
    chan_str = chan_strs{chan_idx};
    filename_in = fullfile(INPUTDIRECTORY,sprintf('%s_round%.03i_%s_registered.tif',FILEROOT_NAME_INPUT,round_num,chan_str));
    fprintf('Loading %s\n',filename_in);
    data = load3DTif_uint16(filename_in);
    data = data(YRANGE,XRANGE,ZRANGE);

    filename_out = fullfile(OUTPUTDIRECTORY,sprintf('%s_round%.03i_%s_registered.tif',FILEROOT_NAME_OUTPUT,round_num,chan_str));
    save3DTif_uint16(data,filename_out);

end

