%% This is more a protocol document than a run script
rajlab_inputdir = '/Users/Goody/Neuro/ExSeq/rajlab/ExSeqCultureNormedRegged/';
cd(rajlab_inputdir);

%% have to do these one at a time, as they are nonblocking GUIS
% ie, DON'T run this as a cell
%draw a rectangle that includes the entire dataset
improc2.segmentGUI.SegmentGUI;

%Then run this line at the command line
%internal rajlab process, this takes about 5-10 minutes to run
improc2.processImageObjects();

%Then, run this to get as many reasonable puncta as you can per rount
improc2.launchThresholdGUI;

%To get the data, use the getPuncta.m file afterward