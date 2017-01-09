function [puncta_deduped] = removeRedundantPuncta(puncta_candidates,DOPLOT,imgDims)

    if nargin==1
        DOPLOT = 0;
    elseif nargin == 2
        disp('Please suppy image dimensions as a parameter');
    end
            

    nbd = 2; %The neighborhood parameter to cluster puncta

    cluster_ids = zeros(1,size(puncta_candidates,1));

    cluster_ctr = 1;

    for puncta_idx = 1:size(puncta_candidates,1)

        %first do a rough filtering of the dataset to get only the regions
        %around a puncta
        puncta_location = puncta_candidates(puncta_idx,:);

        %generate a 2*max(epsilon)+1 size window around the puncta of interest
        y_min = puncta_location(1) - nbd;
        y_max = puncta_location(1) + nbd;
        x_min = puncta_location(2) - nbd;
        x_max = puncta_location(2) + nbd;
        z_min = puncta_location(3) - nbd;
        z_max = puncta_location(3) + nbd;

        candidate_puncta_neighbors = [];

        %find the distances to the other indices
        y_candidates = puncta_candidates(:,1);
        y_indices = y_candidates>=y_min & y_candidates<=y_max;

        x_candidates = puncta_candidates(:,2);
        x_indices = x_candidates>=x_min & x_candidates<= x_max;

        z_candidates = puncta_candidates(:,3);
        z_indices = z_candidates>=z_min & z_candidates <= z_max;

        %Keep the indices that satisfy all three dimension constraints
        otherpuncta_indices = y_indices & x_indices & z_indices;
        %now remove the neighbor that is the original puncta in question
        otherpuncta_indices(puncta_idx) = 0;
        %get the locations of the neighbors
        otherpuncta_neighbors = puncta_candidates(otherpuncta_indices,:);

        %Are there any duplicate points in this neighborhood?
        if size(otherpuncta_neighbors,1)>0
            

            %If the puncta in question is not part of a cluster (common)
            if cluster_ids(puncta_idx)==0
                cluster_ids(otherpuncta_indices) = cluster_ctr;
                cluster_ids(puncta_idx) = cluster_ctr;
                cluster_ctr = cluster_ctr+1;
                %if the puncta in question is part of an existing cluster
            else
                cluster_ids(otherpuncta_indices) = cluster_ids(puncta_idx);
            end
        end

    end


    % Get the indices that correspond to redundant puncta
    cluster_indices = cluster_ids>0;

    if DOPLOT
        figure; hold on;

        scatter(puncta_candidates(:,1),puncta_candidates(:,2),'b.');

        scatter(puncta_candidates(cluster_indices,1),puncta_candidates(cluster_indices,2),'ro');
        xlim([1,imgDims(2)]);
        ylim([1,imgDims(1)]);
    end


    % To handle the duplicates:
    % calculate the mean position, append this to the end of the output list
    % Remove the original locations using the [] tool

    puncta_deduped = puncta_candidates;

    %add the merged duplicates to the output
    for c_id = 1:cluster_ctr-1
        duplicates = puncta_candidates(cluster_ids==c_id,:);
        centroid = mean(duplicates,1);
        puncta_deduped(end+1,:) = centroid;
        if DOPLOT
        scatter(centroid(1),centroid(2),'g*');
        end
    end

    %remove the original redundant points
    puncta_deduped(cluster_indices,:) = [];
    if DOPLOT
        hold off;
    end
end