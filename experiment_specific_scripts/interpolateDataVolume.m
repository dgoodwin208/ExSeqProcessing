function interpolateDataVolume(round_num,Z_upsample_factor)
%interpolateDataVolumeSummary Load data and upsample the data



FILEROOT_NAME_INPUT = 'exseqautoframe7';
FILEROOT_NAME_OUTPUT = 'exseqautoframe7i';

fprintf('Starting processing of round %i\n',round_num);
%Load all channels, normalize them, calculate the cross corr of
%channels 1-3 vs 4

DIRECTORY = '/mp/nas0/ExSeq/AutoSeqHippocampusOrig/4_registration';

chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

for chan_idx = 1:length(chan_strs)
    chan_str = chan_strs{chan_idx};
    filename_in = fullfile(DIRECTORY,sprintf('%s_round%.03i_%s_registered.tif',FILEROOT_NAME_INPUT,round_num,chan_str));
    fprintf('Loading %s\n',filename_in);
    data = load3DTif_uint16(filename_in);
    
    indices = 1:size(data,3);
    query_pts = 1:1/Z_upsample_factor:size(data,3);
    
    data_interp = interp1(indices,squeeze(data(1,1,:)),query_pts);
    data_interpolated = zeros(size(data,1),size(data,2),length(data_interp));
    
    for y = 1:size(data,1)
        for x = 1:size(data,2)
            data_interpolated(y,x,:) = interp1(indices,squeeze(data(y,x,:)),query_pts,'spline');
        end
        
        if mod(y,200)==0
            fprintf('\t%i/%i rows processed\n',y,size(data,1));
        end
    end
    
    filename_out = fullfile(DIRECTORY,sprintf('%s_round%.03i_%s_registered.tif',FILEROOT_NAME_OUTPUT,round_num,chan_str));
    save3DTif_uint16(data_interpolated,filename_out);
    fprintf('Saving %s\n',filename_out);

end
