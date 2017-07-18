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



% We first filter all puncta that are within PUNCTASIZE/2 of a boundary
% Because the images are registered at this point and the puncta locations
% are taken from the params.REFERENCE_ROUND_PUNCTA
bad_puncta_indices = [];
ctr = 1;
for puncta_idx = 1:num_puncta
    y_indices = Y(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Y(puncta_idx) + params.PUNCTA_SIZE/2;
    x_indices = X(puncta_idx) - params.PUNCTA_SIZE/2 + 1: X(puncta_idx) + params.PUNCTA_SIZE/2;
    z_indices = Z(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Z(puncta_idx) + params.PUNCTA_SIZE/2;
    
    if any([x_indices y_indices z_indices]<1) || ...
            any(x_indices>data_width) || any(y_indices>data_height) || any(z_indices>data_depth)
        bad_puncta_indices = union(bad_puncta_indices,puncta_idx);
        continue
    end
    
    %use the meshgrid command to get all the pixels
    [y_grid,x_grid,z_grid] = meshgrid(y_indices,x_indices,z_indices);
    
    x_total_indices_cell(ctr) = {x_grid(:)};
    y_total_indices_cell(ctr) = {y_grid(:)};
    z_total_indices_cell(ctr) = {z_grid(:)};
    
    ctr = ctr +1;
    if mod(puncta_idx,1000)==0
        fprintf('Analyzed %i/%i puncta for spatial constraints\n',...
            puncta_idx,num_puncta);
    end
end

%Use the cell2mat trick to avoid growing the array in the for loop
x_total_indices = cell2mat(x_total_indices_cell);
y_total_indices = cell2mat(y_total_indices_cell);
z_total_indices = cell2mat(z_total_indices_cell);
%just have to linearize it:
x_total_indices = x_total_indices(:);
y_total_indices = y_total_indices(:);
z_total_indices = z_total_indices(:);

%If we want to do any other spatial filtering, do it now.
% X_MIN = 1; X_MAX = data_width;
% Y_MIN = 1; Y_MAX = data_height;
% Z_MIN = 1; Z_MAX = data_depth;

%Remove the indices of the bad puncta that are too close to the boundaries
good_puncta_indices = setdiff(1:num_puncta,bad_puncta_indices);

%Remove all puncta from the set that are too close to the boundary of the
%image
puncta_set = puncta_set(:,:,:,:,:,good_puncta_indices);

%Also keep track of the location of each puncta
Y = Y(good_puncta_indices);
X = X(good_puncta_indices);
Z = Z(good_puncta_indices);

%update our number of puncta:
num_puncta_filtered = length(X);
fprintf('Filtered %i/%i puncta that were too close to the boundaries\n',...
    num_puncta-num_puncta_filtered,num_puncta);
clearvars num_puncta;

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS);
parfor exp_idx = 1:params.NUM_ROUNDS
    disp(['round=',num2str(exp_idx)])
    
    %Load all channels of data into memory for one experiment
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    for c_idx = params.COLOR_VEC
        experiment_set(:,:,:,c_idx) = load3DTif(organized_data_files{exp_idx,c_idx});
    end
    
    disp(['[',num2str(exp_idx),'] processing puncta in parallel'])
    
    puncta_set_cell{exp_idx} = cell(params.NUM_CHANNELS,num_puncta_filtered);
    
    
    for puncta_idx = 1:num_puncta_filtered
        for c_idx = params.COLOR_VEC
            y_indices = Y(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Y(puncta_idx) + params.PUNCTA_SIZE/2;
            x_indices = X(puncta_idx) - params.PUNCTA_SIZE/2 + 1: X(puncta_idx) + params.PUNCTA_SIZE/2;
            z_indices = Z(puncta_idx) - params.PUNCTA_SIZE/2 + 1: Z(puncta_idx) + params.PUNCTA_SIZE/2;
            
            puncta_set_cell{exp_idx}{c_idx,puncta_idx} = experiment_set(y_indices,x_indices,z_indices,c_idx);
        end
    end
end

%%
clear data_cols

disp('reducing processed puncta')
% reduction of parfor
for exp_idx = 1:params.NUM_ROUNDS
    for puncta_idx = 1:num_puncta_filtered
        for c_idx = params.COLOR_VEC
            if ~isequal(size(puncta_set_cell{exp_idx}{c_idx,puncta_idx}), [0 0])
                puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,puncta_idx};
            end
        end
    end
end
clear puncta_set_cell
%%% DOING MINOR ADJUSTMENTS TO PUNCTA LOCATIONS WHILE 
bad_puncta = [];
puncta_fix_cell = cell(size(puncta_set,6),1);
all_shifts = cell(size(puncta_set,6),1);

parfor p_idx = 1:size(puncta_set,6)

    puncta = puncta_set(:,:,:,:,:,p_idx);
    shifts_per_exp = zeros(params.NUM_ROUNDS,3);
    %sum of channels (that have now been quantile normalized)
    puncta_roundref = sum(squeeze(puncta(:,:,:,params.REFERENCE_ROUND_PUNCTA,:)),4);
    offsetrange = [2,2,2];
    %Assuming the first round is reference round for puncta finding
    %Todo: take this out of prototype
    moving_exp_indices = 1:params.NUM_ROUNDS; moving_exp_indices(params.REFERENCE_ROUND_PUNCTA) = [];


    for e_idx = moving_exp_indices
        %Get the sum of the colors for the moving channel
        puncta_roundmove = sum(squeeze(puncta(:,:,:,e_idx,:)),4);
        [~,shifts] = crossCorr3D(puncta_roundref,puncta_roundmove,offsetrange);
        shifts_per_exp(e_idx,:) = shifts;
        if numel(shifts)>3
            %A maximum point wasn't found for this round, likely indicating
            %something weird with the round. Skip this whole puncta
            fprintf('Error in Round %i in Puncta %i\n', e_idx,p_idx);
            bad_puncta = [bad_puncta p_idx];
            puncta_fix_cell{p_idx} = puncta;
            continue;
        end
        
       for c_idx = 1:params.NUM_CHANNELS
            %puncta(:,:,:,e_idx,c_idx) = ...
            %    imtranslate3D(squeeze(puncta(:,:,:,e_idx,c_idx)),shifts);
       
            %calculate a new centroid of the puncta as determined by the offset
            Yctr=Y(p_idx)+shifts(1);
	    Xctr=X(p_idx)+shifts(2);
	    Zctr=Z(p_idx)+shifts(3);
             
            y_indices = Yctr - params.PUNCTA_SIZE/2 + 1: Yctr + params.PUNCTA_SIZE/2;
            x_indices = Xctr - params.PUNCTA_SIZE/2 + 1: Xctr + params.PUNCTA_SIZE/2;
            z_indices = Zctr - params.PUNCTA_SIZE/2 + 1: Zctr + params.PUNCTA_SIZE/2;

            puncta_set_cell{exp_idx}{c_idx,puncta_idx} = experiment_set(y_indices,x_indices,z_indices,c_idx);
            
      end
    
end

   all_shifts{p_idx}= shifts_per_exp
   puncta_fix_cell{p_idx} = puncta;

   if mod(p_idx,1000)==0
       fprintf('3D crosscorr fix for puncta  %i/%i\n',p_idx,size(puncta_set,6));
   end
end
%%% End new minor tweak
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

