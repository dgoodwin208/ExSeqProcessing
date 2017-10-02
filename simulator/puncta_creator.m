% loadSimParams;
%% Generate number of puncta, positions and the transcripts
load('groundtruth_dictionary.mat')


volume_microns = (simparams.IMAGE_FOVSIZE_XY*simparams.IMAGE_RESOLUTION_XY)^2 *...
    (simparams.IMAGE_FOVSIZE_Z*simparams.IMAGE_RESOLUTION_Z);

num_puncta = floor(volume_microns*simparams.VOLUME_DENSITY);

%Create rand positions, one pixel away from any edges
xpos = randi([2 simparams.IMAGE_FOVSIZE_XY-1],num_puncta,1);
ypos = randi([2 simparams.IMAGE_FOVSIZE_XY-1],num_puncta,1);
zpos = randi([2 simparams.IMAGE_FOVSIZE_Z-1],num_puncta,1);

puncta_pos = [ypos,xpos,zpos];
clear xpos ypos zpos; %don't need once they are in the puncta_pos vector
puncta_transcripts = groundtruth_codes(randperm(size(groundtruth_codes,1),num_puncta),:);
clear groundtruth_codes; %don't need once we've loaded the random subset

%add the primer rounds to the transcripts
puncta_transcripts = [ones(num_puncta,1),3*ones(num_puncta,1),2*ones(num_puncta,1), puncta_transcripts];
%Each puncta will will be described in terms of the gaussian covariance
%matrix
puncta_covs = zeros(num_puncta,3);

for p_idx = 1:num_puncta
    %Recall sigma=FWHM/(2*sqrt(2*ln(2)))
    %So we use the parameters to determine the 3D gaussian parameters
    cov_for_puncta = [-1 -1 -1]; %just initialize it for the while loop
    
    while any(cov_for_puncta<.2) %.2 is magic number to avoid funky looking puncta
        %Because of the parameters, there are times when the covariance
        %parameters come in negative, which means the gaussian produces
        %complex numbers :/ In which case we just re-loop for better random
        %values
        cov_for_puncta = normrnd(simparams.PUNCTA_SIZE_MEAN,simparams.PUNCTA_SIZE_STD,1,3)/(2*sqrt(2*log(2)));
    end
    puncta_covs(p_idx,:) = cov_for_puncta;
end

filename_groundtruth=fullfile(simparams.OUTPUTDIR,sprintf('%s_groundtruth_pos+transcripts.mat',simparams.SIMULATION_NAME));
%just save everything, but these varables are most important:
%'puncta_pos','puncta_transcripts','puncta_covs'
save(filename_groundtruth);
%% Use the function across all rounds
for rnd_idx = 1:size(puncta_transcripts,2)
    % Produce the raw output of gaussians across four channels
    % Add 
    [ simulated_data] = makeSimulatedRound(num_puncta,puncta_transcripts(:,rnd_idx),...
        simparams.PUNCTA_CROSSTALK,...
        puncta_pos+normrnd(simparams.PUNCTA_DRIFT_MEAN,simparams.PUNCTA_DRIFT_MEAN,num_puncta,3),...
        puncta_covs,...
        simparams.PUNCTA_SIZE_PRCTCHANGE_ACROSS_ROUNDS,...
        [simparams.IMAGE_FOVSIZE_XY,simparams.IMAGE_FOVSIZE_XY,simparams.IMAGE_FOVSIZE_Z]);
    
    % Scale with intensity of channel, then add background
    simulated_data_scaled = simulated_data;
    
    for c_idx = 1:4
        simulated_data_scaled(:,:,:,c_idx) = simparams.PUNCTA_BRIGHTNESS_MEANS(c_idx)*simparams.MEAN_CORRECTION_FACTOR*simulated_data(:,:,:,c_idx);
        simulated_data_scaled(:,:,:,c_idx) = simparams.CHANNEL_BACKGROUND(c_idx) + simulated_data_scaled(:,:,:,c_idx);
        simulated_data_scaled(:,:,:,c_idx) = normrnd(simparams.MICROSCOPE_NOISE_FLOOR_MEAN,simparams.MICROSCOPE_NOISE_FLOOR_STD,...
            simparams.IMAGE_FOVSIZE_XY,simparams.IMAGE_FOVSIZE_XY,simparams.IMAGE_FOVSIZE_Z)+simulated_data_scaled(:,:,:,c_idx);
    end
    

    % Save this round
    
    
    for c_idx = 1:4
        filename = fullfile(simparams.OUTPUTDIR,sprintf('%s_round%.03i_%s_registered.tif',simparams.SIMULATION_NAME,...
            rnd_idx,simparams.chan_strs{c_idx}));
        save3DTif_uint16(squeeze(simulated_data_scaled(:,:,:,c_idx)),filename);
    end
end

%% Viz tools that were helpful in development:

    % confirm that we're getting four figures of good plot
%     figure;
%     for c_idx = 1:4
%         subplot(2,2,c_idx)
%         imagesc(max(squeeze(simulated_data_scaled(:,:,:,c_idx)),[],3));
%         axis off;
%         title(sprintf('Chan %i',c_idx));
%     end
%     
%     figure;
%     imshow(squeeze(makeRGBImageFrom4ChanData(simulated_data_scaled(:,:,30,:))));
    
