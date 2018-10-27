%% Parameters


dir_input = './';%'/om/user/dgoodwin/ExSeq/';
dir_rootoutput = './input/';
experiment_string = 'sa0916dncv';
if ~exist(dir_rootoutput, 'dir')
    mkdir(dir_rootoutput);
end

files = dir([dir_input '*.ics']);
num_channels = 4;

for file_idx = 1:length(files)
    
    
    datastack = bfopen([dir_input files(file_idx).name]);
    
    img_sample = datastack{1}{1};
    
    
    raw_data = zeros(size(img_sample,1),size(img_sample,2),length(datastack{1}));
    for z=1:length(datastack{1})
        raw_data(:,:,z) = datastack{1}{z};
    end
    
    
    % Assuming four colors and a summed!
    z_stack_size = length(datastack{1})/num_channels;
    
    chan_data= zeros(size(raw_data,1),size(raw_data,2),z_stack_size);
    %The .ics file is organized in z as [chan1],[chan2], etc.
    %Note: this is different than loading with FIJI and saving as tif!
    for c_idx = 1:num_channels    
        for z=1:z_stack_size
            chan_data(:,:,z) = raw_data(:,:,(c_idx-1)*z_stack_size+z);
        end
        save3DTif(chan_data,fullfile(dir_rootoutput,sprintf('%s_round%.03i_chan0%i.tif',...
            experiment_string, file_idx,c_idx)));
    end
    
    clear chan_data
    
    summed = zeros(size(raw_data,1),size(raw_data,2),z_stack_size);
    for z=1:z_stack_size
        %For a particular z slice get all four colors
        c_indices = ([1:num_channels]-1)*z_stack_size+z;
        summed(:,:,z) = sum(raw_data(:,:,c_indices),3);
    end
    save3DTif(summed,fullfile(dir_rootoutput,sprintf('%s_round%i_summed.tif',...
        experiment_string, file_idx)));
    
end

