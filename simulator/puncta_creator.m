
%Define paramters
IMAGE_RESOLUTION_XY = .165; %microns
IMAGE_RESOLUTION_Z = .165; %microns
IMAGE_FOVSIZE_XY = 250;
IMAGE_FOVSIZE_Z = 60;

%puncta size (I think u=10 and s=3.5 is correct but it's slow)
PUNCTA_SIZE_MEAN = 8; %FWHM
PUNCTA_SIZE_STD = 3;
PUNCTA_SIZE_PRCTCHANGE_ACROSS_ROUNDS = .10; %std deviation centered around 100%

%Puncta brightness and crosstalk:
PUNCTA_BRIGHTNESS_MEANS = [800;800;800;300];
% PUNCTA_BRIGHTNESS_STDS = [200;200;200;50];
PUNCTA_CROSSTALK = [[.7 .1 .1 .1];
                    [.1 .7 .1 .1];
                    [.1 .1 .7 .1];
                    [.1 .1 .1 .7];
                    ];


%puncta_spatial distribution (per cubic micron)
VOLUME_DENSITY = 10/100; %puncta per cubic micron

%Offset and background and noise
CHANNEL_BACKGROUND = [100,100,100,100];

SIMULATION_NAME = 'simseqtryone';
OUTPUTDIR = 'simulation_output';

MICROSCOPE_NOISE_FLOOR_MEAN = 10;
MICROSCOPE_NOISE_FLOOR_STD = 5;
%% Generate number of puncta, positions and the transcripts

if ~exist('groundtruth_codes','var')
    loadGroundTruthSequences;
    clear gtlabels; %don't need these for now
end

volume_microns = (IMAGE_FOVSIZE_XY*IMAGE_RESOLUTION_XY)^2 *...
    (IMAGE_FOVSIZE_Z*IMAGE_RESOLUTION_Z);

num_puncta = floor(volume_microns*VOLUME_DENSITY);

%Create rand positions, one pixel away from any edges
xpos = randi([2 IMAGE_FOVSIZE_XY-1],num_puncta,1);
ypos = randi([2 IMAGE_FOVSIZE_XY-1],num_puncta,1);
zpos = randi([2 IMAGE_FOVSIZE_Z-1],num_puncta,1);

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
    while any(cov_for_puncta<.1) %.1 is magic number to avoid funky looking puncta
        %Because of the parameters, there are times when the covariance
        %parameters come in negative, which means the gaussian produces
        %complex numbers :/
        cov_for_puncta = normrnd(PUNCTA_SIZE_MEAN,PUNCTA_SIZE_STD,1,3)/(2*sqrt(2*log(2)));
    end
    puncta_covs(p_idx,:) = cov_for_puncta;
end

filename_groundtruth=fullfile(OUTPUTDIR,sprintf('%s_groundtruth_pos+transcripts.mat',SIMULATION_NAME));
%just save everything, but these varables are most important:
%'puncta_pos','puncta_transcripts','puncta_covs'
save(filename_groundtruth);
%% Use the function across all rounds
for rnd_idx = 1:size(puncta_transcripts,2)
    % Produce the raw output of gaussians across four channels
    [ simulated_data] = makeSimulatedRound(num_puncta,puncta_transcripts(:,rnd_idx),...
        PUNCTA_CROSSTALK,puncta_pos,puncta_covs,PUNCTA_SIZE_PRCTCHANGE_ACROSS_ROUNDS,...
        [IMAGE_FOVSIZE_XY,IMAGE_FOVSIZE_XY,IMAGE_FOVSIZE_Z]);
    
    % Scale with intensity of channel, then add background
    simulated_data_scaled = simulated_data;
    %This is a temporary fix for the fact that the gaussians are produced such
    %that all pixels sum to one, so the max value of simulated data will be
    %small, like ~.17, strongly dependent on the covar matrix
    MEAN_CORRECTION_FACTOR = 400;
    for c_idx = 1:4
        simulated_data_scaled(:,:,:,c_idx) = PUNCTA_BRIGHTNESS_MEANS(c_idx)*MEAN_CORRECTION_FACTOR*simulated_data(:,:,:,c_idx);
        simulated_data_scaled(:,:,:,c_idx) = CHANNEL_BACKGROUND(c_idx) + simulated_data_scaled(:,:,:,c_idx);
        simulated_data_scaled(:,:,:,c_idx) = normrnd(MICROSCOPE_NOISE_FLOOR_MEAN,MICROSCOPE_NOISE_FLOOR_STD,...
            IMAGE_FOVSIZE_XY,IMAGE_FOVSIZE_XY,IMAGE_FOVSIZE_Z)+simulated_data_scaled(:,:,:,c_idx);
    end
    

    % Save this round
    
    chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};
    for c_idx = 1:4
        filename = fullfile(OUTPUTDIR,sprintf('%s_round%.03i_%s.tif',SIMULATION_NAME,...
            rnd_idx,chan_strs{c_idx}));
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
    
