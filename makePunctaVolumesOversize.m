%Load the params file which includes all parameters specific to the
%experiment
loadParameters;

%Get the filtered puncta coords from analyzePuncta.m script
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_filtered.mat',params.FILE_BASENAME)));

%Get the list of all registred files
files = dir(fullfile(params.registeredImagesDir,'*.tif'));

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

%unpack the filtered coords into Y,X,Z vectors
Y = round(puncta_filtered(:,1));
X = round(puncta_filtered(:,2));
Z = round(puncta_filtered(:,3));

num_puncta = length(X); %from the filtered RajLab coordinates

%Define the overly generous size of the puncta for these ROIS
PSIZE = 14; %2*params.PUNCTA_SIZE;

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS,1);
badpuncta_cell = cell(params.NUM_ROUNDS,1);
shifts_cell = cell(params.NUM_ROUNDS,1);

%Now we're ready to loop over all puncta in all other rounds
%And for each puncta in each round, doing the minor puncta adjustments
experiement_indices_for_parallel_loop = 1:params.NUM_ROUNDS;
% For now compare the reference to itself to confirm we get zeros
% experiement_indices_for_parallel_loop(params.REFERENCE_ROUND_PUNCTA) = [];
parfor exp_idx = experiement_indices_for_parallel_loop
    
    disp(['round=',num2str(exp_idx)])
    bad_puncta = [];
    
    %Load all channels of data into memory for one experiment
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    
    for c_idx = params.COLOR_VEC
        experiment_set(:,:,:,c_idx) = load3DTif(organized_data_files{exp_idx,c_idx});
    end
    
    disp(['[',num2str(exp_idx),'] processing puncta in parallel'])
    
    puncta_set_cell{exp_idx} = cell(params.NUM_CHANNELS,num_puncta);
    
    % Loop over every puncta:
    % Compare this round's puncta with the reference round puncta
    % Adjust the indices according to the shift
    % Then store the shift for later analysis
    for puncta_idx = 1:num_puncta
        y_indices = Y(puncta_idx) - PSIZE/2 + 1: Y(puncta_idx) + PSIZE/2;
        x_indices = X(puncta_idx) - PSIZE/2 + 1: X(puncta_idx) + PSIZE/2;
        z_indices = Z(puncta_idx) - PSIZE/2 + 1: Z(puncta_idx) + PSIZE/2;
        
        if any([y_indices(1),x_indices(1),z_indices(1)])<=0
		bad_puncta = [bad_puncta puncta_idx];
                disp('YARP');
		continue;
	 end   
        %Recreate the candidate with the new fixes
        for c_idx = params.COLOR_VEC
            puncta_set_cell{exp_idx}{c_idx,puncta_idx} = experiment_set(y_indices,x_indices,z_indices,c_idx);
        end
        
    end
    
    badpuncta_cell{exp_idx} = bad_puncta;
end




%%
clear data_cols

disp('reducing processed puncta')
puncta_set = zeros(PSIZE,PSIZE,PSIZE, ...
    params.NUM_ROUNDS,params.NUM_CHANNELS,num_puncta);

%reduction of all the bad puncta
total_bad_puncta_indices = [];
for exp_idx = 1:params.NUM_ROUNDS
    total_bad_puncta_indices = [total_bad_puncta_indices; badpuncta_cell{exp_idx}];
end
total_badpuncta = unique(total_bad_puncta_indices);

%Create a list of all the puncta that legit
total_good_puncta = 1:num_puncta;
total_good_puncta(total_badpuncta) = [];
fprintf('Color correction discarded %i puncta\n',length(total_badpuncta));

% reduction of parfor
for exp_idx = 1:params.NUM_ROUNDS
    for puncta_idx = 1:length(total_good_puncta)
        original_puncta_idx = total_good_puncta(puncta_idx);   
        for c_idx = params.COLOR_VEC
            %if ~isequal(size(puncta_set_cell{exp_idx}{c_idx,puncta_idx}), [0 0])
            %    fprintf('Error with exp_idx=%i c_idx=%i puncta_idx=%i\n',exp_idx,c_idx,puncta_idx);
            %end
                puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,original_puncta_idx};
        end
    end
end

%reduction of shifts per puncta
% shifts = zeros(length(total_good_puncta),3,params.NUM_ROUNDS);
% for exp_idx = 1:params.NUM_ROUNDS
%     shifts_all_puncta= shifts_cell{exp_idx};
%     shifts(:,:,exp_idx) = shifts_all_puncta(total_good_puncta);
% end

Y = Y(total_good_puncta);
X = X(total_good_puncta);
Z = Z(total_good_puncta);


clear puncta_set_cell shifts_cell badpuncta_cell

disp('saving files from makePunctaVolumes')
%just save puncta_set
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois_oversize.mat',params.FILE_BASENAME)),...
    'puncta_set','Y','X','Z','-v7.3');

% save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_shifts.mat',params.FILE_BASENAME)),...
%     'shifts','-v7.3');

%save all the used location values
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_pixels_used_for_puncta.mat',params.FILE_BASENAME)),...
    'x_total_indices','y_total_indices','z_total_indices','-v7.3');


%For reporting, create an image that shows all the windows
figfilename = fullfile(params.reportingDir,sprintf('%s_punctasubvolumemaxproj.fig',params.FILE_BASENAME));
fprintf('generating output file %s\n',figfilename);
figure('Visible','off');
mask = zeros(size(data));
indices_punctamask = [y_total_indices x_total_indices z_total_indices];
%We could remove redundant puncta for speed, but I actually think that it
%is slower than simply linearly indexing across all 3D image
%[indices_unique, ia, ic] = unique(indices_punctamask,'rows');
indices_linear = sub2ind(size(data),indices_punctamask(:,1),indices_punctamask(:,2),indices_punctamask(:,3));

mask(indices_linear)=1;
mask = ~mask; %flip to 1=background, 0=puncta subvolme

imagesc(max(mask,[],3));
title('Max Z projection of all puncta subvolumes');
saveas(gcf,figfilename,'fig')


bgdistros_cell = cell(params.NUM_ROUNDS,params.COLOR_VEC);
parfor exp_idx = 1:params.NUM_ROUNDS
    disp(['round=',num2str(exp_idx)])
    
    for c_idx = params.COLOR_VEC
        
        total_data = load3DTif(organized_data_files{exp_idx,c_idx});
        masked_data = total_data(mask(:));
        
        [values,binedges] = histcounts(masked_data(:),params.NUM_BUCKETS);
        
        %Then use our createEmpDistributionFromVector function to make the
        %distributions
        [p,b] = createEmpDistributionFromVector(masked_data(:),binedges);
        
        bgdistros_cell(exp_idx,c_idx) = [p,b];
    end
    
end

save(fullfile(params.punctaSubvolumeDir,sprintf('%s_background_distributions.mat',params.FILE_BASENAME)),...
    'bgdistros_cell','-v7.3');

