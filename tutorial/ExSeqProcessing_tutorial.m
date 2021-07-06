%% Step 0: Put your data in the right file structure
% If you wish to do a rapid test with a small amount of data, run the
% simulator example in analysis/simlator/runSimulatorRound.m

% This is the demonstration from a full-size field of view sample, used in
% the publication Alon et al 2021 https://doi.org/10.1126/science.aax2656
% You can download the 11GB data here: 
% https://drive.google.com/uc?id=1-vnxKfEnQ-TeqQiaQFsWwWEEQsdS68Ky

% Choose a root directory somewhere on your filesystem where you can create
% the necessary file structure. It'll look like this:
% ROOTDIR / 
%           1_deconvolution 
%           2_color-correction
%           3_normalization
%           4_registration
%           5_puncta-extraction
%           6_base-calling

ROOTDIR = ''; 

%Note: the deconvolution directory is a legacy naming. We used to
%deconvolve the raw data, but realized that the benefit was not worth the
%computation time. Copy the downloaded data and put it here
dir_deconv = fullfile(ROOTDIR,'1_deconvolution');
if ~exist(dir_deconv,'dir')
    mkdir(dir_deconv)
end

dir_colorcorr = fullfile(ROOTDIR,'2_color-correction');
if ~exist(dir_colorcorr,'dir')
    mkdir(dir_colorcorr)
end

dir_norm = fullfile(ROOTDIR,'3_normalization');
if ~exist(dir_norm,'dir')
    mkdir(dir_norm)
end

dir_reg = fullfile(ROOTDIR,'4_registration');
if ~exist(dir_reg,'dir')
    mkdir(dir_reg)
end

dir_puncta = fullfile(ROOTDIR,'5_puncta-extraction');
if ~exist(dir_puncta,'dir')
    mkdir(dir_puncta)
end
dir_puncta = fullfile(ROOTDIR,'6_base-calling');
if ~exist(dir_puncta,'dir')
    mkdir(dir_puncta)
end


%% Edit the LoadParameters file 
% While there are many parameters, only a few of them are critical to
% getting the pipeline running for your data. 
% You can view the params struct at any time:
loadParameters;
params
% Make the changes you need directly to the loadParameters file as the
% computing stages will all run loadParameters

%% Run the pipeline

% Step one: downsampling the data (useful for registration) and correct for
% chromatic shifts between the color channels (if any)
stage_downsampling_and_color_correction;

% Step two: combine the color channels, which is useful for registration
% and puncta detection
stage_normalization;

% Step three: register the data to a common coordinate space. You will pick
% a round in params.REFERENCE_ROUND_WARP
stage_registration;

% Step four: identify the puncta and extract all relevant voxels across the
% rounds
stage_puncta_extraction;

% Step five: convert the voxels in the puncta to sequences that can be
% aligned against the dictionary params.GROUND_TRUTH_DICT
% for the demo data, it should be 'groundtruth_dictionary_CZI20190817.mat'
stage_base_calling;