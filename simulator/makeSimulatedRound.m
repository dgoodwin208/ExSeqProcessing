function [ simulated_data] = makeSimulatedRound(num_puncta,bases,cross_talk,puncta_positions,puncta_sizes,puncta_variability,outputDims)
%MAKESIMULATEDROUND Summary of this function goes here
% num_puncta: how many puncta are going into this volume?
% bases: what base is each puncta showing
% cross_talk: what percentage of each signal is in each round?
% puncta_positions: what is the midpoit of the puncta
% puncta_sizes; what are covar numbers for the gausian
% puncta_variability: how much do we modify the size of the puncta
% outputDims: what's the size of the output image

%Note the output is puncta locations and cross talk, but scaling for
%channel intensities etc will be done in a different subroutine

%output:
simulated_data = zeros([outputDims 4]);

%Cell arrays to hold the data so it can be parallelized
puncta_vols = cell(num_puncta,1);
puncta_position_vectors = cell(num_puncta,3);

%Generate the shapes of the puncta for this round
parfor p_idx = 1:num_puncta
    
    %Using the size variability parameter modify the size of the puncta
    %It's a randomly chosen percentage off of 100%. so u=1
    punctasize_modified = normrnd(1,puncta_variability,1,3) .* puncta_sizes(p_idx,:);
    
    %Generate the volume with some padding around it (2.1 is the default)
    gaussObj = nonIsotropicGaussianPSF(punctasize_modified,2.1,'single');
    
    %These objects are a blob with odd size, with symmetric extension on
    %both sides
    y_extent = (size(gaussObj,1)-1)/2;
    x_extent = (size(gaussObj,2)-1)/2;
    z_extent = (size(gaussObj,3)-1)/2;
    
    ypos = puncta_positions(p_idx,1);
    xpos = puncta_positions(p_idx,2);
    zpos = puncta_positions(p_idx,3);
    
    ymin = max(ypos-y_extent,1); ymax = min(ypos+y_extent,outputDims(1));
    xmin = max(xpos-x_extent,1); xmax = min(xpos+x_extent,outputDims(2));
    zmin = max(zpos-z_extent,1); zmax = min(zpos+z_extent,outputDims(3));
    
    %Note the global position for these pixels
    puncta_position_vectors{p_idx}{1} = ymin:ymax;
    puncta_position_vectors{p_idx}{2} = xmin:xmax;
    puncta_position_vectors{p_idx}{3} = zmin:zmax;
    
    %Note the object pixels
    %All the extra indexing is for the case of puncta near the edge of the
    %image
    puncta_vols{p_idx} = gaussObj(...
        ymin - (ypos-y_extent)+1 : end + (ymax - (ypos+y_extent)),...
        xmin - (xpos-x_extent)+1 : end + (xmax - (xpos+x_extent)),...
        zmin - (zpos-z_extent)+1 : end + (zmax - (zpos+z_extent)));
    
    
    
    %[ymin ymax xmin xmax zmin zmax]
    %     [ypos(p_idx) xpos(p_idx) zpos(p_idx)]
    fprintf('%i/%i processed\n',p_idx,num_puncta);
end

%Recollect all the pixels into the final image (in the case of a parfor
%above)
%distribute the signal according to the crosstalk matrix
for p_idx = 1:num_puncta
    
    y_indices = puncta_position_vectors{p_idx}{1};
    x_indices = puncta_position_vectors{p_idx}{2};
    z_indices = puncta_position_vectors{p_idx}{3};
    
    groundtruth_base = bases(p_idx);
    for c_idx = 1:4
        %For each channel, add some percentage of the target fluorescence
        %looping over color channels to add crosstalk where appropriate
        simulated_data(y_indices,x_indices,z_indices,c_idx) = ...
            simulated_data(y_indices,x_indices,z_indices,c_idx) +...
            cross_talk(groundtruth_base,c_idx)* puncta_vols{p_idx};
    end
end



end

