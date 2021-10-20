%% Note all the relevant filepaths

%A few manual parameters to set upfront
% There is a yaml file with all the major parameters of the experiment. We
% will use the name of that yaml file as the experiment_name throughout 
% EXPERIMENT_NAME = 'cruk'; 
%Where do load/save the files for this experiment?
ROOTDIR = '/Users/goody/Neuro/ExSeq/CRUK/';
YAMLNAME = 'cruk.yaml';
DATESTRING = '20210619'; %Can be used as a ID of processes results


global N_READS_MINIMUM N_NOISE_FLOOR SIZE_THRESH CC_CONNECTIVITY
global DOWNSAMPLE_RATE;
%When making the large sample-wide maps, what downsample did we use?
DOWNSAMPLE_RATE = 3; 
N_READS_MINIMUM = 50;
N_NOISE_FLOOR = 0; % All counts per cell at this level or below are set to zero
SIZE_THRESH = 50;%What is the minimum size threshold for a segmented cell?
%Define the more rigorous connectivity to avoid accidental merging of cells
CC_CONNECTIVITY = 4;  % Either 4 or 8


%Define the montage of DAPI? ExSeq reads can be 
imgfile_ds_dapi = fullfile(ROOTDIR,'cruk210226_ds5_round001_ch04.tif');
%Is there a mask of segmented cells? If not, just comment out
imgfile_ds_seg = fullfile(ROOTDIR,'cruk210226_ds5_round001_ch04.vsv.vsseg_export_s0.tif');

yamlfile = fullfile(ROOTDIR,YAMLNAME);
yamlspecs = ReadYaml(yamlfile);

%The YAML files might have extra quotes so remove them
experiment_name = strrep(yamlspecs.basename,'''','');

% Then load these variables into memory

save_dir = sprintf('/Users/goody/Neuro/ExSeq/CRUK/allreads_%s',DATESTRING);
combined_transcriptsfile = fullfile(save_dir,'alltranscripts.mat');
USABLE_HAMMING = 2;


segged_outfile = fullfile(ROOTDIR,sprintf('%s-%s-Manual+TranscriptSeg.mat',experiment_name,DATESTRING) );

%% Run the common code block then
pipeline_downloadAndConsolidateResults;