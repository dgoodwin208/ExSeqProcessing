% Run an end-to-end ExSeq experiment in simulation
% Should take ~5-10 minutes of compute time with default settings.

% The many parameters are managed in these scripts, you are expected to get
% the settings correct
loadSimParams; %only used in the context of simulation
loadParameters; %Used throughout the ExSeqProcessing pipeline

% Puncta simulation can either be done in a grid (to explore arbitrary
% densities) or in random placements. This demo is cur
sim_id = 1; 
% simparams.GRID_XY_SPACING=spacing; 
% simparams.GRID_Z_SPACING=spacing; 

rootdir = '../simseq';


simname = sprintf('simseqtest%i',sim_id);

fov_rootfolder = fullfile(rootdir,simname);

if ~exist(fov_rootfolder,'dir')
    mkdir(fov_rootfolder);
    mkdir(fullfile(fov_rootfolder,'1_deconvolution'));
    mkdir(fullfile(fov_rootfolder,'2_color-correction'));
    mkdir(fullfile(fov_rootfolder,'3_normalization'));
    mkdir(fullfile(fov_rootfolder,'4_registration'));
    mkdir(fullfile(fov_rootfolder,'5_puncta-extraction'));
    mkdir(fullfile(fov_rootfolder,'6_base-calling'));
    mkdir(fullfile(fov_rootfolder,'logs'));
    
end



% NOTE: For now, you will need to use these values to update loadParameters.m
% manually. Soon we will convert everything to a yaml-based system
% The stages of the ExSeqProcessing below rely on calling LoadParameters,
% so you need to update that file with the file folders etc. that you set
% here in this cell.

%Set the parameters for the simulator
simparams.SIMULATION_NAME = simname;
simparams.ROOTDIR = fov_rootfolder;


params.deconvolutionImagesDir = fullfile(fov_rootfolder,'1_deconvolution');
params.colorCorrectionImagesDir = fullfile(fov_rootfolder,'2_color-correction');
params.normalizedImagesDir = fullfile(fov_rootfolder,'3_normalization');
params.registeredImagesDir = fullfile(fov_rootfolder,'4_registration');
params.punctaSubvolumeDir = fullfile(fov_rootfolder,'5_puncta-extraction');
params.basecallingResultsDir = fullfile(fov_rootfolder,'6_base-calling');
params.FILE_BASENAME = simname;


%% Create the simulated data
puncta_creator;

%% Condensed ExSeqProcessing Pipeline called sequentially
setup_cluster_profile();

%If you want to skip the registration step in the simulator, you can use
%the simparams.SAVE_REGISTERED flag. However, skipping registration 
%isn't fully implemented, and you will still need to run the normalization 
%step. We recommend simparams.SAVE_REGISTERED stays false for now.
if ~simparams.SAVE_REGISTERED
    stage_downsampling_and_color_correction;
    stage_normalization;
    stage_registration;
end
stage_puncta_extraction;
stage_base_calling;

%% Load the completed results and examine the performance

%Load the processed results:
load(fullfile(params.basecallingResultsDir,sprintf('%s_transcriptobjects.mat',params.FILE_BASENAME)));
%Get the list of gene names 
insitu_genes_recovered = categorical(cellfun(@(x) x.name,transcript_objects,'UniformOutput',false));

%Load the simulated transcripts
filename_groundtruth=fullfile(simparams.ROOTDIR,sprintf('%s_groundtruth_pos+transcripts.mat',simparams.SIMULATION_NAME));
load(filename_groundtruth,'genes_simulated'); 
genes_simulated = categorical(genes_simulated);

%Load information on the extracted puncta
load(fullfile(params.basecallingResultsDir,sprintf('%s_basecalls.mat',params.FILE_BASENAME)),'puncta_intensities_norm');
n_puncta = size(puncta_intensities_norm,1);
fprintf('%i reads simulated\n',length(genes_simulated));
fprintf('%i puncta recovered\n',n_puncta);
fprintf('%i transcripts aligned\n',length(insitu_genes_recovered));

figure; 
h = histogram(genes_simulated,'DisplayOrder','descend');
hold on;
histogram(insitu_genes_recovered,h.Categories)
title(sprintf('%i reads simulated, %i reads recovered',length(genes_simulated),length(insitu_genes_recovered)));
legend('Simulated','Recovered')




