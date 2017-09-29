loadSimParams;
loadParameters;

%% simseqtwo: 20puncta/100
simparams.SIMULATION_NAME = 'simseqtwo';
simparams.VOLUME_DENSITY = 20/100; %puncta per cubic micron
simparams.OUTPUTDIR = '/mp/nas0/ExSeq/simulator/simseqtwo';
params.registeredImagesDir = '/mp/nas0/ExSeq/simulator/simseqtwo';
params.punctaSubvolumeDir = '/mp/nas0/ExSeq/simulator/simseqtwo';

puncta_creator;
punctafeinder; puncta_filter_exploration;

%% simseqthree: 30puncta/100
simparams.SIMULATION_NAME = 'simseqthree';
simparams.VOLUME_DENSITY = 30/100; %puncta per cubic micron
simparams.OUTPUTDIR = '/mp/nas0/ExSeq/simulator/simseqthree';
params.registeredImagesDir = '/mp/nas0/ExSeq/simulator/simseqthree';
params.punctaSubvolumeDir = '/mp/nas0/ExSeq/simulator/simseqthree';

puncta_creator;
punctafeinder; puncta_filter_exploration;
