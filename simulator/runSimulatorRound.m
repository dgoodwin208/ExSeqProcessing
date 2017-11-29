function [ ] = runSimulatorRound(spacing,rootdir)
%RUNSIMULATORROUND R
loadSimParams;
loadParameters;

simname = sprintf('simseq_spacingtest_%i',spacing);

working_dir = fullfile(rootdir,simname);

if ~exist(working_dir,'dir')
    mkdir(working_dir);
end

%Set the parameters for the simulator
simparams.SIMULATION_NAME = simname;
simparams.OUTPUTDIR = working_dir;

simparams.GRID_XY_SPACING=spacing; 
simparams.GRID_Z_SPACING=spacing; 

params.registeredImagesDir = working_dir;
params.punctaSubvolumeDir = working_dir;
params.FILE_BASENAME = simname;

%Create the puncta
puncta_creator;
number_simulated_puncta = num_puncta; %note this value from the simulator

%Get the ROIs and run naive base caller
% punctafeinder;

%Get the robust puncta paths
% puncta_filter_exploration;

minipipeline;

num_discovered_transcripts = size(unique_transcipts,1);
num_acceptables = sum(final_hammingscores<=1);

output_text = sprintf('num_transcripts=%i\tnum_discovered_transcripts=%i\tnum_acceptable_hamming= %i',...
    number_simulated_puncta,num_discovered_transcripts,num_acceptables);

fileID = fopen(fullfile(working_dir,'results.txt'),'w');
fprintf(fileID,output_text);
fclose(fileID);



end

