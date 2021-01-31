
%Define paramters
simparams.IMAGE_RESOLUTION_XY = .165; %microns
simparams.IMAGE_RESOLUTION_Z = .165; %microns
simparams.IMAGE_FOVSIZE_XY = 150;
simparams.IMAGE_FOVSIZE_Z = 50;

%puncta size (I think u=10 and s=3.5 is correct but it's slow)
simparams.PUNCTA_SIZE_MEAN = 8; %FWHM
simparams.PUNCTA_SIZE_STD = 3;
simparams.PUNCTA_SIZE_PRCTCHANGE_ACROSS_ROUNDS = .10; %std deviation centered around 100%

%puncta position drift (u=2,s=1 is a good guess)
simparams.PUNCTA_DRIFT_MEAN = 0; %2^(1/3); %observed 2 as the total distance in 3d, this params is 1D
simparams.PUNCTA_DRIFT_STD = 0; %.7;

%Puncta brightness and crosstalk:
simparams.PUNCTA_BRIGHTNESS_MEANS = [800;800;800;300];
% PUNCTA_BRIGHTNESS_STDS = [200;200;200;50];
simparams.PUNCTA_CROSSTALK = [[.7 .1 .1 .1];
                    [.1 .7 .1 .1];
                    [.1 .1 .7 .1];
                    [.1 .1 .1 .7];
                    ];


%puncta_spatial distribution (per cubic micron)
simparams.puncta_placement_method='random'; % 'grid' or 'random'
%if simparams.puncta_placement_method ='random', we use this parameter:
simparams.VOLUME_DENSITY = 10/100; %puncta per cubic micron
%if simparams.puncta_placement_method = 'grid', we use these parameters
simparams.GRID_XY_SPACING=10; 
simparams.GRID_Z_SPACING=10; 

% Do we want the data to be saved as registered or raw?
% Can declare either depending on which part of the pipeline you want to
% analyze
simparams.SAVE_REGISTERED = false;  

%Offset and background and noise
simparams.CHANNEL_BACKGROUND = [100,100,100,100];

simparams.SIMULATION_NAME = 'simseqtryone';
simparams.ROOTDIR = '../simulator/sweep/';

simparams.MICROSCOPE_NOISE_FLOOR_MEAN = 10;
simparams.MICROSCOPE_NOISE_FLOOR_STD = 5;

%This is a temporary fix for the fact that the gaussians are produced such
%that all pixels sum to one, so the max value of simulated data will be
%small, like ~.17, strongly dependent on the covar matrix
simparams.MEAN_CORRECTION_FACTOR = 400;

simparams.chan_strs = {'ch00','ch01','ch02','ch03'};
simparams.chan_strs_corr = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};
