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

%Keep track of all x,y,z indices that we use to create the puncta
%subvolumes: We will use all the other locations to create a distribution
%of background values per channel per round
% x_total_indices = [];
% y_total_indices = [];
% z_total_indices = [];


% AnalyzePuncta now checks for closeness to a border, so we have removed
% some of the logic in this next loop
% for puncta_idx = 1:num_puncta
%     y_indices = Y(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Y(puncta_idx) + params.PUNCTA_SIZE/2;
%     x_indices = X(puncta_idx) - params.PUNCTA_SIZE/2 + 1: X(puncta_idx) + params.PUNCTA_SIZE/2;
%     z_indices = Z(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Z(puncta_idx) + params.PUNCTA_SIZE/2;
%
%     %use the meshgrid command to get all the pixels
%     [y_grid,x_grid,z_grid] = meshgrid(y_indices,x_indices,z_indices);
%
%     x_total_indices_cell(puncta_idx) = {x_grid(:)};
%     y_total_indices_cell(puncta_idx) = {y_grid(:)};
%     z_total_indices_cell(puncta_idx) = {z_grid(:)};
%
%
%     if mod(puncta_idx,1000)==0
%         fprintf('Analyzed %i/%i puncta for spatial constraints\n',...
%             puncta_idx,num_puncta);
%     end
% end
%
% %Use the cell2mat trick to avoid growing the array in the for loop
% x_total_indices = cell2mat(x_total_indices_cell);
% y_total_indices = cell2mat(y_total_indices_cell);
% z_total_indices = cell2mat(z_total_indices_cell);
% %just have to linearize it:
% x_total_indices = x_total_indices(:);
% y_total_indices = y_total_indices(:);
% z_total_indices = z_total_indices(:);

%If we want to do any other spatial filtering, do it now.
% X_MIN = 1; X_MAX = data_width;
% Y_MIN = 1; Y_MAX = data_height;
% Z_MIN = 1; Z_MAX = data_depth;

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS);
badpuncta_cell = cell(params.NUM_ROUNDS);
shifts_cell = cell(params.NUM_ROUNDS);
%Before the parallelized loop, create the set of puncta volumes for the
%reference volume, that will then be copied into the different workers for
%calculting the fine adjustments for aligning the pixels
%Load all channels of data into memory for one experiment
exp_idx = params.REFERENCE_ROUND_PUNCTA;
reference_puncta = cell(params.NUM_CHANNELS,num_puncta);
experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);

fprintf('First loading the img volume params.REFERENCE_ROUND_PUNCTA=%i\n',...
    params.REFERENCE_ROUND_PUNCTA);

for c_idx = params.COLOR_VEC
    experiment_set(:,:,:,c_idx) = load3DTif(organized_data_files{exp_idx,c_idx});
end

for puncta_idx = 1:num_puncta_filtered
    for c_idx = params.COLOR_VEC
        y_indices = Y(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Y(puncta_idx) + params.PUNCTA_SIZE/2;
        x_indices = X(puncta_idx) - params.PUNCTA_SIZE/2 + 1: X(puncta_idx) + params.PUNCTA_SIZE/2;
        z_indices = Z(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Z(puncta_idx) + params.PUNCTA_SIZE/2;
        
        reference_puncta{c_idx,puncta_idx} = experiment_set(y_indices,x_indices,z_indices,c_idx);
    end
end

%Then copy this result into the aggregator vairable puncta_set_cell
%Which will be put back together after we loop over rounds colors and
puncta_set_cell{exp_idx} = reference_puncta;
clear experiment_set; %avoid any possible broadcasting of variables in the parfor

%Now we're ready to loop over all puncta in all other rounds
%And for each puncta in each round, doing the minor puncta adjustments
experiement_indices_for_parallel_loop = 1:params.NUM_ROUNDS;
% For now compare the reference to itself to confirm we get zeros
% experiement_indices_for_parallel_loop(params.REFERENCE_ROUND_PUNCTA) = [];
parfor exp_idx = experiement_indices_for_parallel_loop
    
    disp(['round=',num2str(exp_idx)])
    bad_puncta = [];
    shifts = zeros(num_puncta,3);
    offsetrange = [2,2,2];
    
    %Load all channels of data into memory for one experiment
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    
    for c_idx = params.COLOR_VEC
        experiment_set(:,:,:,c_idx) = load3DTif(organized_data_files{exp_idx,c_idx});
    end
    
    disp(['[',num2str(exp_idx),'] processing puncta in parallel'])
    
    puncta_set_cell{exp_idx} = cell(params.NUM_CHANNELS,num_puncta_filtered);
    
    
    % Loop over every puncta:
    % Compare this round's puncta with the reference round puncta
    % Adjust the indices according to the shift
    % Then store the shift for later analysis
    for puncta_idx = 1:num_puncta_filtered
        y_indices = Y(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Y(puncta_idx) + params.PUNCTA_SIZE/2;
        x_indices = X(puncta_idx) - params.PUNCTA_SIZE/2 + 1: X(puncta_idx) + params.PUNCTA_SIZE/2;
        z_indices = Z(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Z(puncta_idx) + params.PUNCTA_SIZE/2;
        
        %Create the candidate puncta
        candidate = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.NUM_CHANNELS);
        %Extract out the reference puncta colors so we can sum them up
        reference_puncta_in_colors = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.NUM_CHANNELS);
        for c_idx = params.COLOR_VEC
            candidate(:,:,:,c_idx) = experiment_set(y_indices,x_indices,z_indices,c_idx);
            reference_puncta_in_colors(:,:,:,c_idx) = reference_puncta{c_idx,puncta_idx}
        end
        
        candidate_sum = sum(candidate,4);
        reference_sum = sum(reference_puncta_in_colors,4);
        
        [~,shifts] = crossCorr3D(reference_sum,candidate_sum,offsetrange);
        if numel(shifts)>3
            %A maximum point wasn't found for this round, likely indicating
            %something weird with the round.
            fprintf('Error in Round %i in Puncta %i\n', e_idx,puncta_idx);
            
            %Take note of this puncta for later - perhaps it makes most
            %sense to discard these puncta preemptively
            bad_puncta = [bad_puncta; puncta_idx];
            
            %Store the non-adjusted data as a placeholder
            for c_idx = params.COLOR_VEC
                puncta_set_cell{exp_idx}{c_idx,puncta_idx} = candidate(:,:,:,c_idx);
            end
            
            continue;
        end
        
        % Use the pixel adjustments:
        y_indices = y_indices + shifts(1);
        x_indices = x_indices + shifts(2);
        z_indices = z_indices + shifts(3);
        
        %Recreate the candidate with the new fixes
        for c_idx = params.COLOR_VEC
            candidate(:,:,:,c_idx) = experiment_set(y_indices,x_indices,z_indices,c_idx);
        end
        
        shifts(puncta_idx,:) = shifts;
        puncta_set_cell{exp_idx}{c_idx,puncta_idx} = candidate;
    end
    
    shifts_cell{exp_idx} = shifts;
    badpuncta_cell{exp_idx} = bad_puncta;
    clear experiment_set
end




%%
clear data_cols

disp('reducing processed puncta')
puncta_set = zeros(params.PUNCTA_SIZE,params.PUNCTA_SIZE,params.PUNCTA_SIZE, ...
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
    for puncta_idx = total_good_puncta
        
        for c_idx = params.COLOR_VEC
            if ~isequal(size(puncta_set_cell{exp_idx}{c_idx,puncta_idx}), [0 0])
                fprintf('Error with exp_idx=%i c_idx=%i puncta_idx=%i\n',exp_idx,c_idx,puncta_idx);
            end
                puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,puncta_idx};
        end
    end
end

%reduction of shifts per puncta
shifts = zeros(num_puncta,3,params.NUM_ROUNDS);
for exp_idx = 1:params.NUM_ROUNDS
    shifts(:,:,exp_idx) = shifts_cell{exp_idx};
end

Y = Y(total_good_puncta);
X = X(total_good_puncta);
Z = Z(total_good_puncta);


clear puncta_set_cell shifts_cell badpuncta_cell

disp('saving files from makePunctaVolumes')
%just save puncta_set
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)),...
    'puncta_set','Y','X','Z','-v7.3');

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

