%% Note all the relevant filepaths
%Set these variables

DATESTRING = '20210510'; %Can be used as a ID of processes results

%Define the montage of DAPI? ExSeq reads can be 
imgfile_ds_dapi = '/Users/goody/Neuro/ExSeq/HTAPP_917/htapp917_ds4_round002_ch04.tif';
%Is there a mask of segmented cells? If not, just comment out
%imgfile_ds_seg = '/Users/goody/Neuro/ExSeq/HTAPP_917/HTAPP_917.vsseg_export_s0.tif';

yamlfile = '/Users/goody/Neuro/ExSeq/HTAPP Common/htapp_upload/htapp_%i.yaml';
yamlspecs = ReadYaml(yamlfile);

%The YAML files might have extra quotes so remove them
experiment_name = strrep(yamlspecs.basename,'''','');

% Then load these variables into memory
DOWNSAMPLE_RATE = 4; %This is what's been used for HTAPP sampes

save_dir = sprintf('/Users/goody/Neuro/ExSeq/HTAPP_%i/allreads_%s',EXP_NUM,DATESTRING);
combined_transcriptsfile = fullfile(save_dir,sprintf('htapp%i-alltranscripts.mat',EXP_NUM));
%Note the total maps are downampled by a factor of 2!
OUTDIR = sprintf('/Users/goody/Neuro/ExSeq/HTAPP_%i/',EXP_NUM);

%% Run the common code block then
pipeline_downloadAndConsolidateResults;