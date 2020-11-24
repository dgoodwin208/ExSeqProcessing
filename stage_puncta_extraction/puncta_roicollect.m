loadParameters;
filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids,'puncta_centroids','puncta_voxels','crop_dims')
%% Collect the subvolumes we started this with, but now only with the pixels from the puncta!

num_insitu_transcripts = size(puncta_voxels,1);

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS,1);
puncta_indices_cell = cell(params.NUM_ROUNDS,1);
%the puncta indices are here in linear form for a specific round
delete(gcp('nocreate'));
conditions = conditions_for_concurrency();
max_pool_size = concurrency_size_in_puncta_roicollect(conditions);
fprintf('PUNCTA_MAX_POOL_SIZE=%d\n', max_pool_size);
parpool(max_pool_size); %arbitrary but this parallel loop is memory intensive

run_num_list = 1:params.NUM_ROUNDS;
if isfield(params, 'MORPHOLOGY_ROUND') && (params.MORPHOLOGY_ROUND <= params.NUM_ROUNDS)
    run_num_list(params.MORPHOLOGY_ROUND) = [];
end
%IN BRANCH: Adding in the newly cropped feature 
didLoadCropDims = exist('crop_dims','var');
parfor exp_idx = run_num_list
    disp(['round=',num2str(exp_idx)])
    pixels_per_rnd = []; pixels_per_rnd_bg = []; %Try to clear memory
    %clear pixels_per_rnd pixels_per_rnd_bg; 
    pixels_per_rnd = cell(num_insitu_transcripts,params.NUM_CHANNELS);
    pixindices_per_rnd = cell(num_insitu_transcripts,1); 
    hasNotedIndices = false;

    for c_idx = params.COLOR_VEC
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_%s.%s',params.FILE_BASENAME,exp_idx,params.SHIFT_CHAN_STRS{c_idx},regparams.REGISTRATION_TYPE,params.IMAGE_EXT));
        img =  load3DImage_uint16(filename_in);
        %IN BRANCH: Adding in the newly cropped feature
        if didLoadCropDims 
            img = img(...
                crop_dims(1,1):crop_dims(1,2),...
                crop_dims(2,1):crop_dims(2,2),...
                crop_dims(3,1):crop_dims(3,2));
        end
        for puncta_idx = 1:num_insitu_transcripts
           
            indices_for_puncta = puncta_voxels{puncta_idx};
            
            if ~hasNotedIndices
                 pixindices_per_rnd{puncta_idx} = indices_for_puncta;
            end
            
            %Get all the pixel intensity values for the puncta
            pixels_per_rnd{puncta_idx,c_idx}= uint16(img(indices_for_puncta));
            
            %clear img_subregion imgmask_subregion background_pixels;       
            if mod(puncta_idx,10000)==0
                fprintf('Rnd %i, Chan %i, Puncta %i processed\n',exp_idx,c_idx,puncta_idx);
            end
            
        end
        hasNotedIndices=true;
    end
    puncta_set_cell{exp_idx} = pixels_per_rnd;
    puncta_indices_cell{exp_idx} = pixindices_per_rnd;
end


disp('reducing processed puncta')
puncta_set_median = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
puncta_set_max = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
puncta_set_mean = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
% reduction of parfor
for puncta_idx = 1:num_insitu_transcripts
    for exp_idx = run_num_list
        for c_idx = params.COLOR_VEC
            % Each puncta_set_cell per exp is
            % pixels_per_rnd = cell(num_insitu_transcripts,params.NUM_CHANNELS);
            pixel_vector = puncta_set_cell{exp_idx}{puncta_idx,c_idx};
            
            
            puncta_set_median(exp_idx,c_idx,puncta_idx) = median(pixel_vector);
            puncta_set_max(exp_idx,c_idx,puncta_idx) = max(pixel_vector);
            [vals, indices] = sort(pixel_vector,'descend');
            %only take the average of the top N voxels, where N is the minimum size threshold. This may be helpful in the case of larger puncta that get some background into the segmentation of the puncta
            puncta_set_mean(exp_idx,c_idx,puncta_idx) = mean(vals(1:params.PUNCTA_SIZE_THRESHOLD));
           
        end
    end
end

if isfield(params, 'MORPHOLOGY_ROUND') && (params.MORPHOLOGY_ROUND <= params.NUM_ROUNDS)
    puncta_set_median(params.MORPHOLOGY_ROUND,:,:) = [];
    puncta_set_max(params.MORPHOLOGY_ROUND,:,:) = [];
    puncta_set_mean(params.MORPHOLOGY_ROUND,:,:) = [];
end


% Using median values to ignore bad points
%Puncta set median is NUM_ROUNDS x NUM_CHANNELS x NUM_PUNCTA
channels_notpresent = squeeze(sum(puncta_set_median==0,2));
%channels_notpresent is NUM_ROUNDS x NUM Puncta where each value is the
%number of zero-valued channels in that round and puncta

%This is worth exploring (DG 09/20/2018). In principle, we only need one channel present, and in the case of high passing the data we might be missing a few channels
%Was this line:
%num_roundsmissing_per_puncta = squeeze(sum(channels_notpresent>2,1));

num_roundsmissing_per_puncta = squeeze(sum(channels_notpresent>3,1));

%Create a mask of all puncta that have a non-zero signal in all four
%channels for all rounds
%When we high pass the signal before this step, there are a lot of zeros
%so we can be more lenient. We allow two channels to be empty now, meaning
%the pixel value was less than the low passed value

%signal_complete = num_roundsmissing_per_puncta==0;
%Working to keep puncta even when missing up to one round
signal_complete = num_roundsmissing_per_puncta<=params.MAXNUM_MISSINGROUND;

fprintf('Number of complete puncta: %i \n',sum(signal_complete));

%Apply the signal complete filter before saving the raw data
for exp_idx = run_num_list
    puncta_set_cell{exp_idx} = puncta_set_cell{exp_idx}(signal_complete,:);
    puncta_indices_cell{exp_idx} = puncta_indices_cell{exp_idx}(signal_complete);
end

if isfield(params, 'MORPHOLOGY_ROUND') && (params.MORPHOLOGY_ROUND <= params.NUM_ROUNDS)
    puncta_set_cell(params.MORPHOLOGY_ROUND) = [];
    puncta_indices_cell(params.MORPHOLOGY_ROUND) = [];
end

outputfile = fullfile(params.punctaSubvolumeDir,sprintf('%s_punctavoxels.mat',params.FILE_BASENAME));
%IN BRANCH: In the interim 
if ~didLoadCropDims 
    crop_dims = false;
else
    save(outputfile,'puncta_set_cell','puncta_indices_cell','crop_dims','-v7.3');
end


