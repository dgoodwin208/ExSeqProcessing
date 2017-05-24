%Load the params file which includes all parameters specific to the
%experiment
loadParameters;

%Get the filtered puncta coords from analyzePuncta.m script
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_filtered.mat',params.FILE_BASENAME)));

%Get the list of all registred files
files = dir(fullfile(params.registeredImagesDir,'*.tif'));

%The convertMicroscope files currently takes the alphabetical order
%Which is not the right ordering of actual experiments
% round_correction_indices = [10,11,12,1,2,3,4,5,6,7,8,9];

organized_data_files = cell(params.NUM_ROUNDS,params.NUM_CHANNELS);

chan_list = {};
for file_index = 1:length(files)
    
    %Only load the chan datasets, can ignore the summed for now
    if strfind(files(file_index).name,'summed')
        continue;
    end
    
    %Need to crop out round number and channel
    %FULLTPSsa0916dncv_round7_chan1.tif

    m = regexp(files(file_index).name,'.+_round(\d+)_(\w+).tif','tokens');
    round_num = str2num(m{1}{1});
    chan_name = m{1}{2};
    if sum(ismember(chan_list,chan_name)) == 0
        chan_num = length(chan_list)+1;
        chan_list{chan_num} = chan_name;
    else
        chan_num = find(ismember(chan_list,chan_name));
    end

    organized_data_files{round_num,chan_num} = fullfile(params.registeredImagesDir,files(file_index).name);
end

%% 
%For each experiment, load the four channels of data, then
%start getting the subregions

data = load3DTif(organized_data_files{1,1});
data_height = size(data,1);
data_width = size(data,2);
data_depth = size(data,3);

%unpack the filtered coords into X,Y,Z vectors
Y = round(puncta_filtered(:,1));
X = round(puncta_filtered(:,2));
Z = round(puncta_filtered(:,3));

num_puncta = length(X); %from the RajLab coordinates 

%Keep track of all x,y,z indices that we use to create the puncta
%subvolumes: We will use all the other locations to create a distribution
%of background values per channel per round
x_total_indices = [];
y_total_indices = [];
z_total_indices = [];

%Create the whole size of the puncta_set vector optimistically: not all the
%puncta will be within PUNCTA_SIZE of a boundary, in which we case we will
%not use that puncta. That means there will be some empty puncta which we
%need to remove 
puncta_set = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE, ...
    params.NUM_ROUNDS,params.NUM_CHANNELS,num_puncta);

bad_puncta_indices = [];

x_total_indices_cell = cell(params.NUM_ROUNDS);
y_total_indices_cell = cell(params.NUM_ROUNDS);
z_total_indices_cell = cell(params.NUM_ROUNDS);
puncta_set_cell = cell(params.NUM_ROUNDS);
bad_puncta_indices_cell = cell(params.NUM_ROUNDS);

parfor exp_idx = 1:params.NUM_ROUNDS
    disp(['round=',num2str(exp_idx)])

    %Load all channels of data into memory for one experiment
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    for c_idx = params.COLOR_VEC
       experiment_set(:,:,:,c_idx) = load3DTif(organized_data_files{exp_idx,c_idx});
    end
    
    disp(['[',num2str(exp_idx),'] processing puncta in parallel')
    x_total_indices_cell{exp_idx} = [];
    y_total_indices_cell{exp_idx} = [];
    z_total_indices_cell{exp_idx} = [];
    puncta_set_cell{exp_idx} = cell(params.NUM_CHANNELS,num_puncta);
    bad_puncta_indices_cell{exp_idx} = [];


    %For each puncta in each channel in each round, get the volume
    %PROVIDING that the puncta is not within PUNCTASIZE/2 of a boundary
    for puncta_idx = 1:num_puncta
        for c_idx = params.COLOR_VEC
            y_indices = Y(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Y(puncta_idx) + params.PUNCTA_SIZE/2;
            x_indices = X(puncta_idx) - params.PUNCTA_SIZE/2 + 1: X(puncta_idx) + params.PUNCTA_SIZE/2;
            z_indices = Z(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Z(puncta_idx) + params.PUNCTA_SIZE/2;
            if any([x_indices y_indices z_indices]<1)
                bad_puncta_indices_cell{exp_idx} = union(bad_puncta_indices_cell{exp_idx},puncta_idx);
                continue
            end
            if any(x_indices>data_width) || any(y_indices>data_height) || any(z_indices>data_depth)
                bad_puncta_indices_cell{exp_idx} = union(bad_puncta_indices_cell{exp_idx},puncta_idx);
                continue
            end
            
            x_total_indices_cell{exp_idx} = [x_total_indices_cell{exp_idx}; x_indices'];
            y_total_indices_cell{exp_idx} = [y_total_indices_cell{exp_idx}; y_indices'];
            z_total_indices_cell{exp_idx} = [z_total_indices_cell{exp_idx}; z_indices'];
            
            puncta_set_cell{exp_idx}{c_idx,puncta_idx} = experiment_set(y_indices,x_indices,z_indices,c_idx);
        end
    end
end

%%
clear data_cols, clear data

disp('reducing processed puncta')
% reduction of parfor
for exp_idx = 1:params.NUM_ROUNDS
    x_total_indices = [x_total_indices; x_total_indices_cell{exp_idx}];
    y_total_indices = [y_total_indices; y_total_indices_cell{exp_idx}];
    z_total_indices = [z_total_indices; z_total_indices_cell{exp_idx}];
    bad_puncta_indices = union(bad_puncta_indices,bad_puncta_indices_cell{exp_idx});

    for puncta_idx = 1:num_puncta
        for c_idx = params.COLOR_VEC
            if ~isequal(size(puncta_set_cell{exp_idx}{c_idx,puncta_idx}), [0 0])
                puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,puncta_idx};
            end
        end
    end
end

good_puncta_indices = setdiff(1:num_puncta,bad_puncta_indices);


%Remove all puncta from the set that are too close to the boundary of the
%image
puncta_set = puncta_set(:,:,:,:,:,good_puncta_indices);

%Also keep track of the location of each puncta
Y = Y(good_puncta_indices);
X = X(good_puncta_indices);
Z = Z(good_puncta_indices);

disp('saving files')
%just save puncta_set
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)),...
    'puncta_set','Y','X','Z','-v7.3');

%save all the used location values
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_pixels_used_for_puncta.mat',params.FILE_BASENAME)),...
    'x_total_indices','y_total_indices','z_total_indices','-v7.3');
