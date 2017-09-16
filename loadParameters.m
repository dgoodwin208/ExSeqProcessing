% Data loading a storage parameteres
params.deconvolutionImagesDir = '/Users/Goody/Neuro/ExSeq/simulator/images';
params.colorCorrectionImagesDir = '/Users/Goody/Neuro/ExSeq/simulator/images';
params.registeredImagesDir = '/Users/Goody/Neuro/ExSeq/simulator/images';
params.punctaSubvolumeDir = '/Users/Goody/Neuro/ExSeq/simulator/puncta';
params.transcriptResultsDir = '/Users/Goody/Neuro/ExSeq/simulator/puncta';

params.FILE_BASENAME = 'simseqtryone';

%Experimental parameters
params.REFERENCE_ROUND_WARP=5;
params.REFERENCE_ROUND_PUNCTA = 5;
params.NUM_ROUNDS = 20;
params.NUM_CHANNELS = 4;
params.PUNCTA_SIZE = 10; %Defines the cubic region around each puncta

%RajLab filtering parameters:
params.PUNCTA_PRESENT_THRESHOLD = 17; %Number of rounds to agree on 
params.PUNCTA_SIZE_THRESHOLD = 10; %Number of rounds to agree on 
params.EPSILON_TARGET = 4; %Radius of neighborhood for puncta to agree across rounds

%Base calling parameters
params.COLOR_VEC = [1,2,3,4]; %Which channels are we comparing? (in case of empty chan)
params.DISTANCE_FROM_CENTER = 2.5; %how far from the center of the puncta subvol?

params.THRESHOLD_EXPRESSION = 15; %If a transcript shows up fewer than this it's probably noise 

%Reporting directories
params.reportingDir = '/home/dgoodwin/ExSeqProcessing/logs/imgs';
