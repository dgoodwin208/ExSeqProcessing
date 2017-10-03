loadSimParams;
loadParameters;

%% simseqtwo: 20puncta/100
simparams.SIMULATION_NAME = 'simseqfive';
simparams.VOLUME_DENSITY = 70/100; %puncta per cubic micron
simparams.OUTPUTDIR = '/mp/nas0/ExSeq/simulator/simseqfive';
params.registeredImagesDir = '/mp/nas0/ExSeq/simulator/simseqfive';
params.punctaSubvolumeDir = '/mp/nas0/ExSeq/simulator/simseqfive';
params.FILE_BASENAME = 'simseqfive';
puncta_creator;
punctafeinder; puncta_filter_exploration;

%% simseqthree: 30puncta/100
simparams.SIMULATION_NAME = 'simseqthree';
simparams.VOLUME_DENSITY = 30/100; %puncta per cubic micron
simparams.OUTPUTDIR = '/mp/nas0/ExSeq/simulator/simseqthree';
params.registeredImagesDir = '/mp/nas0/ExSeq/simulator/simseqthree';
params.punctaSubvolumeDir = '/mp/nas0/ExSeq/simulator/simseqthree';
params.FILE_BASENAME = 'simseqthree';
puncta_creator;
punctafeinder; puncta_filter_exploration;
