% Data loading a storage parameteres
params.registeredImagesDir = '/om/project/boyden/ExSeqSlice/output';
params.rajlabDirectory = '/Users/Goody/Neuro/ExSeq/rajlab/ExSeqSliceNormedBeforeReg/';
params.punctaSubvolumeDir = '/Users/Goody/Neuro/ExSeq/rajlab/ExSeqSliceNormedBeforeReg/';
params.transcriptResultsDir = '';

params.FILE_BASENAME = 'sa0916slicedncv';

%Experimental parameters
params.NUM_ROUNDS = 12;
params.NUM_CHANNELS = 4;
params.PUNCTA_SIZE = 10; %Defines the cubic region around each puncta
%For the case in ExSeq when Dan saved the round number incorrectly
%ToDo: remove this 
params.round_correction_indices = [10,11,12,1,2,3,4,5,6,7,8,9];
% params.round_correction_indices = 1:params.NUM_ROUNDS;

%RajLab filtering parameters:
params.THRESHOLD = 7; %Number of rouds to agree on 
params.EPSILON_TARGET = 4; %Radius of neighborhood for puncta to agree across rounds


%Base calling parameters
params.COLOR_VEC = [1,2,3,4]; %Which channels are we comparing? (in case of empty chan)
params.DISTANCE_FROM_CENTER = 2.5; %how far from the center of the puncta subvol?

params.THRESHOLD_EXPRESSION = 15; %If a transcript shows up fewer than this it's probably noise 
params.NUM_BUCKETS = 500; %For stastical analysis
