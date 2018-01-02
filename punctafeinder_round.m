
function centroids = punctafeinder_round(round_num)

    loadParameters;

    %centroids are the location
    centroids = {};

    fgnd_cell = {};
    stack_cell = {};

    chan_strs = params.CHAN_STRS;
    chan_minimums = [110 150 140 60];
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
        backfind_stack_in = stack_in;
        backfind_stack_in(backfind_stack_in<chan_minimums(chan_num))=Inf;
        
        background = min(backfind_stack_in,[],3);
        backfind_stack_in = [];
        background(isinf(background))=chan_minimums(chan_num);
        
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
            if size(candidate_puncta(i).PixelIdxList,1)< puncta_size_threshold ...
                || size(candidate_puncta(i).PixelIdxList,1)>150
                indices_to_remove = [indices_to_remove i];
            end
        end

        good_indices = 1:length(candidate_puncta);
        good_indices(indices_to_remove) = [];

        filtered_puncta = candidate_puncta(good_indices);
        fprintf('Round%i, Chan %i: removed %i candidate puncta for being too small or large\n',...
            round_num,chan_num,length(candidate_puncta)-length(filtered_puncta));

        centroids_temp = zeros(length(filtered_puncta),3);
        voxels_temp = cell(length(filtered_puncta),1);
        for p_idx = 1:length(filtered_puncta)
            centroids_temp(p_idx,:) = filtered_puncta(p_idx).WeightedCentroid;
            voxels_temp{p_idx} = filtered_puncta(p_idx).PixelIdxList;
        end

        centroids{round_num, chan_num} = filtered_puncta;

        output_img = zeros(img_size{chan_num});

        for i= 1:length(filtered_puncta)
            %Set the pixel value to somethign non-zero
            output_img(filtered_puncta(i).PixelIdxList)=100;
        end
        merged_puncta = [];

        filename_out = fullfile(punctaSubvolumeDir,sprintf('%s_round%.03i_%s_puncta.tif',basename,round_num,chan_strs{chan_num}));
        save3DTif_uint16(output_img,filename_out);
        output_img = [];
        toc

    end
    fprintf('[%03d] end parfor - puncta merge\n',round_num);

end
