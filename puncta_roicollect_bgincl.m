%loadParameters;

filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroids,'puncta_centroids','puncta_voxels')
%% Collect the subvolumes we started this with, but now only with the pixels from the puncta!

num_insitu_transcripts = size(puncta_voxels,1);

%Define a puncta_set object that can be parallelized
puncta_set_cell = cell(params.NUM_ROUNDS,1);
puncta_set_cell_bgmean = cell(params.NUM_ROUNDS,1);
puncta_set_cell_bgmedian = cell(params.NUM_ROUNDS,1);
BGREGION_SEARCHXY = 30;
BGREGION_SEARCHZ = 5;
%the puncta indices are here in linear form for a specific round

parpool(3); %arbitrary but this parallel loop is memory intensive
filename_punctaMask = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
img_mask = load3DImage_uint16(filename_punctaMask)>0;
    
parfor exp_idx = 1:params.NUM_ROUNDS 
    disp(['round=',num2str(exp_idx)])
    pixels_per_rnd = []; pixels_per_rnd_bg = []; %Try to clear memory
    %clear pixels_per_rnd pixels_per_rnd_bg; 
    pixels_per_rnd = cell(num_insitu_transcripts,params.NUM_CHANNELS);
    pixels_per_rnd_bgmean = cell(num_insitu_transcripts,params.NUM_CHANNELS);
    pixels_per_rnd_bgmedian = cell(num_insitu_transcripts,params.NUM_CHANNELS);

    for c_idx = params.COLOR_VEC
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_%s.%s',params.FILE_BASENAME,exp_idx,params.CHAN_STRS{c_idx},regparams.REGISTRATION_TYPE,params.IMAGE_EXT));
        img =  load3DImage_uint16(filename_in);
        %Leftover from an experiment of turning each image into a DFF calculation
        %img_blur = imgaussfilt3(single(img),[30 30 30*(params.XRES/params.ZRES)]); 
        %imgdff = (img-img_blur)./(img_blur);
        %img = max(imgdff,0); 
        for puncta_idx = 1:num_insitu_transcripts
            
            indices_for_puncta = puncta_voxels{puncta_idx};
            
            %Get all the pixel intensity values for the puncta
            pixels_per_rnd{puncta_idx,c_idx}= img(indices_for_puncta);
             
            %Get all the neighborhood pixels from the background
            y_min = floor(max(1,puncta_centroids(puncta_idx,2)-BGREGION_SEARCHXY+1));
            y_max = floor(min(size(img_mask,1),puncta_centroids(puncta_idx,2)+BGREGION_SEARCHXY));
            x_min = floor(max(1,puncta_centroids(puncta_idx,1)-BGREGION_SEARCHXY+1));
            x_max = floor(min(size(img_mask,2),puncta_centroids(puncta_idx,1)+BGREGION_SEARCHXY));
            z_min = floor(max(1,puncta_centroids(puncta_idx,3)-BGREGION_SEARCHZ+1));
            z_max = floor(min(size(img_mask,3),puncta_centroids(puncta_idx,3)+BGREGION_SEARCHZ));
            
            img_subregion = img(y_min:y_max,x_min:x_max,z_min:z_max);
            imgmask_subregion = img_mask(y_min:y_max,x_min:x_max,z_min:z_max);
            background_pixels = img_subregion(~imgmask_subregion);
            
            pixels_per_rnd_bgmean{puncta_idx,c_idx} = mean(background_pixels);
            pixels_per_rnd_bgmedian{puncta_idx,c_idx} = median(background_pixels);
            
            %iclear img_subregion imgmask_subregion background_pixels;       
            if mod(puncta_idx,10000)==0
                fprintf('Rnd %i, Chan %i, Puncta %i processed\n',exp_idx,c_idx,puncta_idx);
            end
            
        end
        
    end
    puncta_set_cell{exp_idx} = pixels_per_rnd;
    puncta_set_cell_bgmean{exp_idx} = pixels_per_rnd_bgmean;
    puncta_set_cell_bgmedian{exp_idx} = pixels_per_rnd_bgmedian;
end

disp('reducing processed puncta')
puncta_set_median = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
puncta_set_max = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
puncta_set_mean = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
puncta_set_backgroundmean= zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
puncta_set_backgroundmedian= zeros(params.NUM_ROUNDS,params.NUM_CHANNELS,num_insitu_transcripts);
% reduction of parfor
for puncta_idx = 1:num_insitu_transcripts
    for exp_idx = 1:params.NUM_ROUNDS
        for c_idx = params.COLOR_VEC
            % Each puncta_set_cell per exp is
            % pixels_per_rnd = cell(num_insitu_transcripts,params.NUM_CHANNELS);
            pixel_vector = puncta_set_cell{exp_idx}{puncta_idx,c_idx};
            
            pixel_vector_bgmean = puncta_set_cell_bgmean{exp_idx}{puncta_idx,c_idx};
            pixel_vector_bgmedian = puncta_set_cell_bgmedian{exp_idx}{puncta_idx,c_idx}; 
            puncta_set_backgroundmedian(exp_idx,c_idx,puncta_idx) = pixel_vector_bgmedian;
            puncta_set_backgroundmean(exp_idx,c_idx,puncta_idx) = pixel_vector_bgmean;
            
            puncta_set_median(exp_idx,c_idx,puncta_idx) = median(pixel_vector);
            puncta_set_max(exp_idx,c_idx,puncta_idx) = max(pixel_vector);
            [vals, indices] = sort(pixel_vector,'descend');
            %only take the average of the top N voxels, where N is the minimum size threshold. This may be helpful in the case of larger puncta that get some background into the segmentation of the puncta
            puncta_set_mean(exp_idx,c_idx,puncta_idx) = mean(vals(1:params.PUNCTA_SIZE_THRESHOLD));
           
        end
    end
end

% Using median values to ignore bad points
%channels_present = squeeze(sum(puncta_set_median==0,2));
%num_roundspresent_per_puncta = squeeze(sum(channels_present==0,1));
%Create a mask of all puncta that have a non-zero signal in all four
%channels for all rounds
%signal_complete = num_roundspresent_per_puncta==params.NUM_ROUNDS;

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
signal_complete = num_roundsmissing_per_puncta==0;

fprintf('Number of complete puncta: %i \n',sum(signal_complete));

puncta_set_median = puncta_set_median(:,:,signal_complete);
puncta_set_max = puncta_set_max(:,:,signal_complete);
puncta_set_mean = puncta_set_mean(:,:,signal_complete);
puncta_set_backgroundmean = puncta_set_backgroundmean(:,:,signal_complete);
puncta_set_backgroundmedian = puncta_set_backgroundmedian(:,:,signal_complete);

puncta_centroids = puncta_centroids(signal_complete,:);


indices_of_good_puncta = find(signal_complete);
puncta_voxels = puncta_voxels(indices_of_good_puncta);

outputfile = fullfile(params.transcriptResultsDir,sprintf('%s_puncta_pixels.mat',params.FILE_BASENAME));
save(outputfile,'puncta_set_median','puncta_set_max','puncta_set_mean',...
    'puncta_centroids','puncta_voxels','puncta_set_backgroundmean',...
    'puncta_set_backgroundmedian','-v7.3');

