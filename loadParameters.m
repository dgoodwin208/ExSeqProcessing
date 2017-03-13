% Data loading a storage parameteres
params.registeredImagesDir = '/om/project/boyden/ExSeqCulture/output';
params.rajlabDirectory = '/Users/Goody/Neuro/ExSeq/rajlab/splintr16bNormed/';
params.punctaSubvolumeDir = '/Users/Goody/Neuro/ExSeq/rajlab/splintr16bNormed/';
params.transcriptResultsDir = '';

%Experimental parameters
params.NUM_ROUNDS = 3;
params.NUM_CHANNELS = 4;
params.PUNCTA_SIZE = 10; %Defines the cubic region around each puncta
%For the case in ExSeq when Dan saved the round number incorrectly
%ToDo: remove this 
%params.round_correction_indices = [10,11,12,1,2,3,4,5,6,7,8,9];
params.round_correction_indices = 1:params.NUM_ROUNDS;

%RajLab filtering parameters:
params.THRESHOLD = 2; %Number of rouds to agree on 
params.EPSILON_TARGET = 4; %Radius of neighborhood for puncta to agree across rounds


%Base calling parameters
params.COLOR_VEC = [1,2,4]; %Which channels are we comparing? (in case of empty chan)
params.DISTANCE_FROM_CENTER = 2.5; %how far from the center of the puncta subvol?


params.NUM_BUCKETS = 500; %For stastical analysis
