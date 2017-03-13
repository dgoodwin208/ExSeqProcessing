function [ sumloglikelihood ] = calculateJointProbability( datavec, distro_vals, distro_bins)
%CALCULATEJOINTPROBABILITY Calculate the log likelihood of the datavec
%being drawn from the given distro

% Todo, probably should change the function name ;)

%As a cheap lookup for now (can interp later if necessary), for each entry
%that we're trying to find the vector, simply subtract it and the min. 
%Hacky but fast!
%https://www.mathworks.com/matlabcentral/newsreader/view_thread/268230
% prob_vals = zeros(size(datavec));
% for i = 1:length(datavec)
%     %Find the index of the probability value by finding the nearest entry
%     %in the distro_bins vector
%     [~, index] = min(abs(distro_bins-datavec(i)));
%     prob_vals(i) = distro_vals(index); % Finds first one only!
%     
% end
%Is this how we use interp
prob_vals = interp1(distro_bins,distro_vals,datavec,'pchip');

%But do a check to make sure no weirdness with the interpolation 
%goes negative
prob_vals(prob_vals<0)=0;

sumloglikelihood = sum(log(prob_vals));
end

