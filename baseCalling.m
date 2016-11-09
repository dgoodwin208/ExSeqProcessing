
%% 

dir_input = '/om/project/boyden/ExSeqSlice/output';

%Get the puncat coords from getPuncta.m function 
load(fullfile(dir_input,'puncta_coords.mat'));

files = dir(fullfile(dir_input,'*.tif'));

NUM_ROUNDS = 12;
NUM_CHANNELS = 4;

%load a sample dataset to get the dimensions
data = load3DTif(fullfile(dir_input,files(1).name));


%The convertMicroscope files currently takes the alphabetical order
%Which is not the right ordering of actual experiments
round_correction_indices = [10,11,12,1,2,3,4,5,6,7,8,9];

%Create a list of objects

datavols = [];
organized_data_files = cell(NUM_ROUNDS,NUM_CHANNELS);

for file_index = 1:length(files)
    
    %Only load the chan datasets
    if findstr(files(file_index).name,'summed')
        continue;
    end
    
    %Need to crop out round number and channel
    %FULLTPSsa0916dncv_round7_chan1.tif
    %Get the latter part of the string, then use sscanf to extract the two
    %ints
    string_parts = split(files(file_index).name,'_round');
    string_parts = split(string_parts{2},'_chan');
    round_num = str2num(string_parts{1});
    
    newdatavol.name = string_parts{1}; 
    
    string_parts = split(string_parts{2},'.');
    chan_num = str2num(string_parts{1});
    
    
    corrected_round_num = round_correction_indices(round_num);
    
    organized_data_files{corrected_round_num,chan_num} = fullfile(dir_input,files(file_index).name);
    
    datavols = [datavols newdatavol];
    %load into a large data structure
end

%% 
%For each experiment, load the four channels of data, normalize them, then
%start getting the subregions

data = load3DTif(organized_data_files{1,1});
data_height = size(data,1);
data_width = size(data,2);
data_depth = size(data,3);

PUNCTA_SIZE = 10;
num_puncta = length(X); %from the RajLab coordinates 

puncta_set = zeros(PUNCTA_SIZE,PUNCTA_SIZE,PUNCTA_SIZE,NUM_ROUNDS,NUM_CHANNELS,num_puncta);

bad_puncta_indices = [];
for exp_idx = 1:NUM_ROUNDS

    exp_idx 
    experiment_set = zeros(data_height,data_width,data_depth, NUM_CHANNELS);
    data_cols = zeros(data_height*data_width*data_depth,NUM_CHANNELS);
    
    %Load the data
    for c_idx = 1:NUM_CHANNELS
       experiment_set(:,:,:,c_idx) = load3DTif(organized_data_files{exp_idx,c_idx});
       data_cols(:,c_idx) = reshape(experiment_set(:,:,:,c_idx),[],1);
    end

%     %Normalize the data
    data_cols = quantilenorm(data_cols);
    
    %Reinsert the data
    for c_idx = 1:NUM_CHANNELS
       experiment_set(:,:,:,c_idx) = reshape(data_cols(:,c_idx),data_height,data_width,data_depth);
    end
    
    %For each puncta in each channel in each round, get the volume
    
    for puncta_idx = 1:num_puncta
        for c_idx = 1:NUM_CHANNELS
            y_indices = Y(puncta_idx) - PUNCTA_SIZE/2 + 1: Y(puncta_idx) + PUNCTA_SIZE/2;
            x_indices = X(puncta_idx) - PUNCTA_SIZE/2 + 1: X(puncta_idx) + PUNCTA_SIZE/2;
            z_indices = Z(puncta_idx) - PUNCTA_SIZE/2 + 1: Z(puncta_idx) + PUNCTA_SIZE/2;
            if any([x_indices y_indices z_indices]<1)
%                 disp('Skipping an out of bounds index');
                bad_puncta_indices = union(bad_puncta_indices,puncta_idx);
                continue
            end
            if any(x_indices>data_width) || any(y_indices>data_height) || any(z_indices>data_depth)
%                 disp('Skipping an out of bounds index');
                bad_puncta_indices = union(bad_puncta_indices,puncta_idx);
                continue
            end
            
            puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = experiment_set(y_indices,x_indices,z_indices,c_idx);
        end
    end
end

%%
clear data_cols, clear data

good_puncta_indices = setdiff(1:num_puncta,bad_puncta_indices);

save(fullfile(dir_input,'roi_parameters_and_punctaset.mat'));

