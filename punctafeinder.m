function punctafeinder()

    loadParameters;

    cluster = parcluster('local_200workers');

    tic;
    disp('===== create batch jobs')

    max_running_jobs = params.PUNCTA_JOB_SIZE;
    waiting_sec = 10;
    in_parallel = true;

    total_round_num = params.NUM_ROUNDS;
    jobs = cell(1,total_round_num);
    running_jobs = zeros(1,total_round_num);
    roundnum = 1;

    %centroids are the location
    centroids = {};

    while roundnum <= total_round_num || sum(running_jobs) > 0
        if (roundnum <= total_round_num) && (sum(running_jobs) < max_running_jobs)
            fprintf('create batch (%d)\n',roundnum)
            if in_parallel
                running_jobs(roundnum) = 1;
                jobs{roundnum} = batch(cluster,@punctafeinder_round,1,{roundnum},'Pool',4,'CaptureDiary',true);
            else
                centroids_round = punctafeinder_round(roundnum);
                for c_idx = 1:params.NUM_CHANNELS
                    centroids{roundnum,c_idx} = centroids_round{roundnum,c_idx};
                end
            end
            roundnum = roundnum+1;
        else
            for job_id = find(running_jobs==1)
                job = jobs{job_id};
                is_finished = 0;
                if strcmp(job.State,'finished')
                    fprintf('batch (%d) has %s.\n',job_id,job.State);
                    output = fetchOutputs(job);
                    centroids_job = output{1};
                    for c_idx = 1:params.NUM_CHANNELS
                        centroids{job_id,c_idx} = centroids_job{job_id,c_idx};
                    end
                    diary(job,['./matlab-puncta-extraction-',num2str(job_id),'.log']);
                    running_jobs(job_id) = 0;
                    delete(job)
                    is_finished = 1;
                elseif strcmp(job.State,'failed')
                    fprintf('batch (%d) has %s, resubmit it.\n',job_id,job.State);
                    diary(job,['./matlab-puncta-extraction-',num2str(job_id),'-failed.log']);
                    jobs{job_id} = recreate(job);
                end
            end
            if is_finished == 0
              disp(['waiting... # of jobs = ',num2str(length(find(running_jobs==1))),'; ',num2str(find(running_jobs==1))])
              pause(waiting_sec);
            end
        end
    end

    disp('===== all batch jobs finished')
    toc;

    makeCentrodsAndVoxels(centroids);

end

function centroids = punctafeinder_round(round_num)

    loadParameters;

    %centroids are the location
    centroids = {};

    fgnd_cell = {};
    stack_cell = {};

    chan_strs = params.CHAN_STRS;
    registeredImagesDir = params.registeredImagesDir;
    basename = params.FILE_BASENAME;
    img_size = {};

    fprintf('[%03d] start parfor - dog\n', round_num);
    parfor chan_num = 1:params.NUM_CHANNELS
        tic
        chan_str = chan_strs{chan_num};
        filename_in = fullfile(registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',basename,round_num,chan_str));
        stack_in = load3DTif_uint16(filename_in);
        img_size{chan_num} = size(stack_in);

        %Todo: trim the registration (not relevant in the crop)
        min_z = round(.1*size(stack_in,3));
        max_z = round(.9*size(stack_in,3));
        background = min(squeeze(stack_in(:,:,min_z:max_z)),[],3);

        toc
        tic
        %stack_original = stack_in;
        stack_in = dog_filter(stack_in);

        %min project of 3D image
        back_dog = dog_filter2d(background);
        background = [];
        %avoding registration artifacts
        %2* is a magic number that just works
        back_dogmax = 2*max(max(back_dog(5:end-5,5:end-5,:))); % avoid weird edge effects
        back_dog = [];

        fgnd_mask = zeros(size(stack_in));
        fgnd_mask(stack_in>back_dogmax) = 1; % use first slice to determine threshold for dog
        fgnd_mask = logical(fgnd_mask); % and get mask

        stack_in(~fgnd_mask) = 0; % thresholded using dog background


        %max project pxls
        %z = -Inf(size(stack_in));
        %z(fgnd_mask) = zscore(single(stack_original(fgnd_mask)));
        fgnd_cell{chan_num} = fgnd_mask;
        stack_cell{chan_num} = stack_in;
        %z_cell{chan_num} = z;
        % max project normalized stuff; after setting bkgd to 0
        toc
    end
    fprintf('[%03d] end parfor - dog\n', round_num);

    %logical OR all foregrounds together
    allmask = fgnd_cell{1} | fgnd_cell{2} | fgnd_cell{3} | fgnd_cell{4};

    %initializig the array of size of the 3d image
    z_cell{1} = -Inf(size(stack_cell{1}));
    z_cell{2} = -Inf(size(stack_cell{2}));
    z_cell{3} = -Inf(size(stack_cell{3}));
    z_cell{4} = -Inf(size(stack_cell{4}));

    %calculate the zscore of all the foreground pixels (done across channels),
    %done per channel
    z_cell{1}(allmask) = zscore(single(stack_cell{1}(allmask)));
    z_cell{2}(allmask) = zscore(single(stack_cell{2}(allmask)));
    z_cell{3}(allmask) = zscore(single(stack_cell{3}(allmask)));
    z_cell{4}(allmask) = zscore(single(stack_cell{4}(allmask)));

    clear allmask;

    %re-masking foreground, now used per channel.
    z_cell{1}(~fgnd_cell{1}) = -Inf;
    z_cell{2}(~fgnd_cell{2}) = -Inf;
    z_cell{3}(~fgnd_cell{3}) = -Inf;
    z_cell{4}(~fgnd_cell{4}) = -Inf;

    clear fgnd_cell;

    %M is the mask
    m = {};

    %Create a new mask per channel based on when a channel is the winner
    [m{1},m{2},m{3},m{4}] = maxprojmask(z_cell{1}, z_cell{2}, z_cell{3}, z_cell{4});

    clear z_cell;

    puncta_size_threshold = params.PUNCTA_SIZE_THRESHOLD;
    punctaSubvolumeDir = params.punctaSubvolumeDir;
    fprintf('[%03d] start parfor - watershed\n',round_num);
    candidate_puncta_cell = {};
    parfor chan_num = 1:params.NUM_CHANNELS
        tic

        stack_in = stack_cell{chan_num};

        % max project

        %set nonlargest to 0
        stack_in(~m{chan_num}) = 0;
        neg_masked_image = -int32(stack_in);
        neg_masked_image(~stack_in) = inf;
        toc
        tic
        L = uint32(watershed(neg_masked_image));
        L(~stack_in) = 0;
        fprintf('wshed\n');
        stack_in = [];
        neg_masked_image = [];
        toc

        filename_in = fullfile(registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',basename,round_num,chan_strs{chan_num}));
        img = load3DTif_uint16(filename_in);

        candidate_puncta_cell{chan_num} = regionprops(L,img, 'WeightedCentroid', 'PixelIdxList');
        L = [];
        img = [];
    end
    fprintf('[%03d] end parfor - watershed\n',round_num);
    stack_cell = {};
    m = {};


    fprintf('[%03d] start parfor - puncta merge\n',round_num);
    parfor chan_num = 1:params.NUM_CHANNELS
        tic
        candidate_puncta = candidate_puncta_cell{chan_num};
        indices_to_remove = [];
        for i= 1:length(candidate_puncta)
            if size(candidate_puncta(i).PixelIdxList,1)< puncta_size_threshold
                indices_to_remove = [indices_to_remove i];
            end
        end

        good_indices = 1:length(candidate_puncta);
        good_indices(indices_to_remove) = [];

        filtered_puncta = candidate_puncta(good_indices);
        fprintf('Round%i, Chan %i: removed %i candidate puncta for being too small\n',...
            round_num,chan_num,length(candidate_puncta)-length(filtered_puncta));

        centroids_temp = zeros(length(filtered_puncta),3);
        voxels_temp = cell(length(filtered_puncta),1);
        for p_idx = 1:length(filtered_puncta)
            centroids_temp(p_idx,:) = filtered_puncta(p_idx).WeightedCentroid;
            voxels_temp{p_idx} = filtered_puncta(p_idx).PixelIdxList;
        end
        toc
        tic

        %Now we need to merge accidental any accidental splits
        XYSEARCH_DISTANCE = 3; %how close to
        ZMAXDISTANCE_FOR_SPLIT = 15;
        fprintf('Running de-splitting where appropriate\n');
        this_round2D = centroids_temp(:,1:2);

        %For all the centroids, find ones that are close in XY
        %For each puncta, find it's nearest neighbor in the same round
        [IDX,D] = knnsearch(this_round2D,this_round2D,'K',5); %getting four other options

        %For each puncta, ignore the mapping to itself, and note the number of
        %possible merge mistakes for this puncta
        merge_pairs = [];
        for puncta_idx = 1:size(this_round2D,1)

            %Find the indices of the D and IDX cols that point to puncta that
            %are within the MERGE_DISTANCE AND not the same puncta
            indices_merge_candidates = (IDX(puncta_idx,:) ~= puncta_idx) & ...
                (D(puncta_idx,:)<=XYSEARCH_DISTANCE);

            if sum(indices_merge_candidates)>0

                merge_candidate_idxs = IDX(puncta_idx,indices_merge_candidates);

                z_distances = centroids_temp(puncta_idx,3) - ...
                    centroids_temp(merge_candidate_idxs,3);

                mergable_candidates_for_voxel_testing = merge_candidate_idxs(abs(z_distances)<ZMAXDISTANCE_FOR_SPLIT);

                %Get the puncta locations
                [y_ref, x_ref,z_ref] = ind2sub(img_size{chan_num},voxels_temp{puncta_idx});

                for mergable_idx = 1:length(mergable_candidates_for_voxel_testing)

                    [y_cand,x_cand,z_cand] = ind2sub(img_size{chan_num},voxels_temp{mergable_candidates_for_voxel_testing(mergable_idx)});

                    if abs(max(z_ref)-min(z_cand))<=2 || abs(max(z_cand)-min(z_ref))<=2

                        %Put the merge pairs in sorted order so it's easier
                        %to remove the duplicates afterward:
                        sorted_pair = sort([puncta_idx mergable_candidates_for_voxel_testing(mergable_idx)]);
                        merge_pairs = [merge_pairs; sorted_pair];
                    end
                end


            end
        end
        toc
        tic


        %merge_pairs will have duplicates, so remove them here
        [unique_pairs, ~, ~] = unique(merge_pairs,'rows');

        %To handle the case of multiple splits in Z, condense the unique
        %puncta into a cell array of combos
        unique_indices = unique(unique_pairs(:));
        counts = sum(unique_pairs(:)==unique_pairs(:)');

        examined_puncta = [];
        merge_combos = {}; combo_ctr = 1;

        %Loop over all of the unique puncta indices that we have found need
        %to be mereged. Find their location in the the unique_pairs matrix
        %and create the merge_combo object
        for u_idx = 1:length(unique_indices)

            last_count_of_neighbors=0;
            unpacked_neighbors = [unique_indices(u_idx)];

%             if unique_indices(u_idx)==1462
%                 barf()
%             end

            if ismember(unpacked_neighbors,examined_puncta)
%                 fprintf('Already examined %i\n',unique_indices(u_idx));
                continue
            end

            %Loop over to make sure all chains of nearby puncta are merged
            while(length(unpacked_neighbors) ~= last_count_of_neighbors)
                last_count_of_neighbors = length(unpacked_neighbors);
                for puncta_query = unpacked_neighbors 
                    %Find all the rows of unique_pairs that have this puncta
                    rows = union(find(unique_pairs(:,1)==puncta_query),find(unique_pairs(:,2)==puncta_query));            
                    unpacked_neighbors = unique([unpacked_neighbors,reshape(unique_pairs(rows,:),1,[])]);
                end

            end

            merge_combos{combo_ctr} = sort(unpacked_neighbors);
            combo_ctr = combo_ctr+1;
            examined_puncta = [examined_puncta, unpacked_neighbors];
        end
        toc
        tic



        merged_puncta = filtered_puncta;
        filtered_puncta = [];
        indices_to_discard = [];
        for combo_idx = 1:length(merge_combos)

            merge_indices = merge_combos{combo_idx};

            N = zeros(length(merge_indices),1);
            total_pixel_list = [];
            centroids_to_merge = zeros(length(merge_indices),3);
            summed_centroid = zeros(3,1);

            for n_idx = 1:length(merge_indices)
                merge_idx = merge_indices(n_idx);

                N(n_idx) = length(merged_puncta(merge_idx).PixelIdxList);
                centroids_to_merge(n_idx,:) = merged_puncta(merge_idx).WeightedCentroid;

                summed_centroid = summed_centroid + N(n_idx)*centroids_to_merge(n_idx,:)';

                total_pixel_list = [total_pixel_list;merged_puncta(merge_idx).PixelIdxList];

            end

            target_merge = merge_indices(1); 
            indices_to_discard = [indices_to_discard, merge_indices(2:end)]; 
            %Combine the pixels by moving the second into first
            merged_puncta(target_merge).PixelIdxList = total_pixel_list;

            %Take a weighted average of the centroids
            merged_puncta(target_merge).WeightedCentroid = summed_centroid/sum(N);

            %for debug purposes, round the positions 

%            for n_idx = 1:length(merge_indices)
%                centroidDebug = round(centroids_to_merge(n_idx,:));
%%                 fprintf('Centroid %i w/ %i pixels [%i %i %i]\t',    ...
%%                      merge_indices(n_idx),N(n_idx),centroidDebug(1),centroidDebug(2),centroidDebug(3));
%            end
%
%            merged_centroidDebug = round( merged_puncta(target_merge).WeightedCentroid);
%%             fprintf('Merge: [%i %i %i]\n',...
%%                 merged_centroidDebug(1),merged_centroidDebug(2),merged_centroidDebug(3));
        end
        toc
        tic

        fprintf('Merging %i out of %i total puncta.\n',length(indices_to_discard),length(merged_puncta));

        %And remove the second idx
        merged_puncta(indices_to_discard) = [];
        indices_to_discard = [];

        centroids{round_num, chan_num} = merged_puncta;

        output_img = zeros(img_size{chan_num});

        for i= 1:length(merged_puncta)
            %Set the pixel value to somethign non-zero
            output_img(merged_puncta(i).PixelIdxList)=100;
        end
        merged_puncta = [];

        filename_out = fullfile(punctaSubvolumeDir,sprintf('%s_round%.03i_%s_puncta.tif',basename,round_num,chan_strs{chan_num}));
        save3DTif_uint16(output_img,filename_out);
        output_img = [];
        toc

    end
    fprintf('[%03d] end parfor - puncta merge\n',round_num);

end

function makeCentrodsAndVoxels(centroids)
    loadParameters;

    %combine the centroid objects into a single set of centroids+pixels per
    %round
    %%
    puncta_centroids = cell(params.NUM_ROUNDS,1);
    puncta_voxels = cell(params.NUM_ROUNDS,1);
    puncta_baseguess = cell(params.NUM_ROUNDS,1);

    for rnd_idx=1:params.NUM_ROUNDS
        num_puncta_per_round = 0;
        for c_idx = 1:params.NUM_CHANNELS
            num_puncta_per_round = num_puncta_per_round + numel(centroids{rnd_idx,c_idx});
        end

        %initialize the vectors for the particular round
        centroids_per_round = zeros(num_puncta_per_round,3);
        centroids_chan_per_round = zeros(num_puncta_per_round,1);
        voxels_per_round = cell(num_puncta_per_round,1);

        ctr = 1;
        for c_idx = 1:params.NUM_CHANNELS
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

