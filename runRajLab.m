%% This is more a protocol document than a run script
rajlab_inputdir = '/Users/Goody/Neuro/ExSeq/rajlab/ExSeqSliceRegged/';
% rajlab_inputdir = '/Users/Goody/Neuro/ExSeq/rajlab/ExSeqCulture/';
cd(rajlab_inputdir);

%% have to do these one at a time, as they are nonblocking GUIS
improc2.segmentGUI.SegmentGUI;
improc2.processImageObjects();
improc2.launchThresholdGUI;

%To get the data, use the getPuncta.m file afterward