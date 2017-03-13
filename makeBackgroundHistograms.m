loadParameters;

load(fullfile(params.punctaSubvolumeDir,'pixels_used_for_puncta.mat'));

% x_total_indices
% y_total_indices
% z_total_indices
% Make a giant Nx3 vector of all the indices
% convert them to 1D using sub2ind
% Then linearize the image and take the values we haven't specified

%Assuming all the incoming data are loaded as integers (which they should
%be)

%Load every file in the registered images directory
%For each file, make a histogram of all data NOT in the total_indices

