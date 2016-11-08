
    
    
tools = improc2.launchImageObjectTools;
out = [];
while tools.iterator.continueIteration
    if (~tools.annotations.getValue('isGood'))
        tools.iterator.goToNextObject;
        continue;
    end
    
    objNum = tools.navigator.currentArrayNum();
    [Y X Z] = tools.objectHandle.getData('alexa').getSpotCoordinates();
    out = [out; Y X Z ones(length(X),1)*objNum];
    tools.iterator.goToNextObject()
end

save('puncta_coords.mat','X','Y','Z');
%% 
dir_input = './cropOutput/';%'/om/user/dgoodwin/ExSeq/';

% TOP_LEFT = [381,147];
% BOTTOM_RIGHT = [390,156];
% Z_RANGE = 49-5+1:49+5;
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
%     newdatavol = datavol; %initialize an 
    
    %Only load the chan datasets
    if findstr(files(file_index).name,'summed')
        continue;
    end
    
    %Need to crop out round number and channel
    %CROPTPSsa0916dncv_round7_chan1.tif
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

save('roi_parameters_and_punctaset.mat');
subplot(NUM_ROUNDS,NUM_CHANNELS,1)


%%
good_puncta_indices = setdiff(1:num_puncta,bad_puncta_indices);
filename = 'non-normalized.gif';
%Setting the figure shape so it comes out well in the gif
set(0, 'DefaultFigurePaperPosition', [425   980   576   876]);
figure(1);
for puncta_idx = good_puncta_indices(1:100)
    subplot_idx = 1;
    for exp_idx = 1:NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));

        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        values = zeros(4,1);
        for c_idx = 1:NUM_CHANNELS

            clims = [min_intensity,max_intensity];
            subplot(NUM_ROUNDS,NUM_CHANNELS,subplot_idx)
            data = squeeze(punctaset_perround(:,:,:,c_idx));
            imagesc(max(data,[],3),clims);

            axis off; colormap gray
            subplot_idx = subplot_idx+1;
        end
    end
%     pause
    
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if puncta_idx == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end