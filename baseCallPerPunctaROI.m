%Base call per round

%output is givin geach pun
loadParameters;

filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));

load(filename_centroids)

%%

%The normalization method is now:
%For each color in each expeirmental round,
%subtract the minimum pixel value,
%then calculate + divide by the mean

%As a proof of concept, make an image that shows those puncta
filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,1,'ch00'));
sample_img = load3DTif_uint16(filename_in);

data_height = size(sample_img,1);
data_width = size(sample_img,2);
data_depth = size(sample_img,3);

puncta_minimums = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
puncta_means = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
puncta_stds = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);

chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);

for exp_idx = 1:params.NUM_ROUNDS
    
    %Get all puncta indices from the punctafeinder
    
    experiment_set = zeros(data_height,data_width,data_depth, params.NUM_CHANNELS);
    disp(['[',num2str(exp_idx),'] loading files'])
    
    for c_idx = params.COLOR_VEC
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,exp_idx,chan_strs{c_idx}));
        experiment_set(:,:,:,c_idx) = load3DTif_uint16(filename_in);
    end
    
    
    puncta_voxel_array_per_round = puncta_voxels{exp_idx};
    puncta_voxel_1D_vector = [];
    for puncta_idx = 1:length(puncta_voxel_array_per_round)
        puncta_voxel_1D_vector = [puncta_voxel_1D_vector; puncta_voxel_array_per_round{puncta_idx}];
    end
    
    for c_idx = params.COLOR_VEC
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s.tif',params.FILE_BASENAME,exp_idx,chan_strs{c_idx}));
        img = load3DTif_uint16(filename_in);
        
        %Get all the pixels from all the puncta for a round and color
        chan_col = img(puncta_voxel_1D_vector); %puncta_set(:,:,:,exp_idx,c_idx,:),[],1);
        
        puncta_minimums(exp_idx,c_idx) = min(chan_col);
        
        chan_col_minshift = chan_col - puncta_minimums(exp_idx,c_idx);
        
        puncta_means(exp_idx,c_idx) = mean(chan_col_minshift);
        puncta_stds(exp_idx,c_idx) = std(chan_col_minshift);
        
    end
    
    
    exp_idx
end

%%

puncta_basecalls = cell(params.NUM_ROUNDS,1);
puncta_baseconfidences = cell(params.NUM_ROUNDS,1);


for exp_idx = 1:params.NUM_ROUNDS
    
    num_puncta = length(puncta_voxels{exp_idx});
    puncta_basecalls_round = zeros(num_puncta,1);
    puncta_baseconfidences = zeros(num_puncta,1);
    
    for puncta_idx = 1:num_puncta
        %Load a 10x10x10 x NUM_CHANNELS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));
        
        %Get the non-zero pixels from the subvolume
        normalized_puncta_vector = cell(params.NUM_CHANNELS,1);
        
        %Get the mask across all rounds:
        punctaset_maxproj_on_channels = max(punctaset_perround,[],4);
        puncta_mask_linear  = punctaset_maxproj_on_channels(:)>0;
        
        normed_mean_values = zeros(4,1);
        for c_idx = 1:params.NUM_CHANNELS
            %load and linearize the pixels
            pixels_from_subvolume = reshape(punctaset_perround(:,:,:,c_idx),1,[]);
            pixels_masked = pixels_from_subvolume(puncta_mask_linear);
            
            %Removing minimum value then dividing by the mean
            %             normalized_puncta_vector{c_idx} = ...
            %                 (pixels_masked - puncta_minimums(exp_idx,c_idx))/puncta_means(exp_idx,c_idx);
            %Standard Z-score method: subtract min, subtract mean, divide by std.
            normalized_puncta_vector{c_idx} = ...
                (pixels_masked - puncta_minimums(exp_idx,c_idx) - puncta_means(exp_idx,c_idx))/puncta_stds(exp_idx,c_idx);
            
            normed_mean_values(c_idx) = mean(normalized_puncta_vector{c_idx});
        end
        
        [values,indices] = sort(normed_mean_values,'descend');
        
        answer_vector(exp_idx) = indices(1);
        
        confidence_vector(exp_idx) = values(1)/(values(1)+values(2));
        
    end
    
    
    
    
end




