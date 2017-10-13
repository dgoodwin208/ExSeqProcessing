%This sample parfor script is just varying spacing 
xyz_spacings = 4:10;
%rootdir = '/mp/nas0/ExSeq/simulator/';
rootdir = '/Users/Goody/Neuro/ExSeq/simulator/sweep';

parfor spacing = xyz_spacings
    runSimulatorRound(spacing,rootdir);
end