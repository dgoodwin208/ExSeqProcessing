function makeCentroidsAndVoxels(centroids)
    loadParameters;

    %combine the centroid objects into a single set of centroids+pixels per
    %round
    %%
    puncta_centroids = cell(params.NUM_ROUNDS,1);
    puncta_voxels = cell(params.NUM_ROUNDS,1);
    puncta_baseguess = cell(params.NUM_ROUNDS,1);

    num_channels = params.NUM_CHANNELS;
    parfor rnd_idx=1:params.NUM_ROUNDS
        num_puncta_per_round = 0;
        for c_idx = 1:num_channels
            num_puncta_per_round = num_puncta_per_round + numel(centroids{rnd_idx,c_idx});
        end

        %initialize the vectors for the particular round
        centroids_per_round = zeros(num_puncta_per_round,3);
        centroids_chan_per_round = zeros(num_puncta_per_round,1);
        voxels_per_round = cell(num_puncta_per_round,1);

        ctr = 1;
        for c_idx = 1:num_channels
            round_objects = centroids{rnd_idx,c_idx};
            for r_idx = 1:size(round_objects,1)
                %Note the centroid of the puncta
                centroids_per_round(ctr,:) = round_objects(r_idx).WeightedCentroid;
                %Note which round it was present in from the punctafeinder
                centroids_chan_per_round(ctr) = c_idx;
                %Note the indices of this puncta
                voxels_per_round{ctr} = round_objects(r_idx).PixelIdxList;
                ctr = ctr +1;
            end
        end

        puncta_centroids{rnd_idx} = centroids_per_round;
        puncta_voxels{rnd_idx} = voxels_per_round;
        puncta_baseguess{rnd_idx} = centroids_chan_per_round;
    end


    filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
    save(filename_centroids,'puncta_centroids','puncta_voxels','puncta_baseguess','-v7.3');

end
