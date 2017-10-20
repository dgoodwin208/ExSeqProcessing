% Post analysis of the new pipeline

%% Get the distances between each tracked puncta per round  

%% Generate an image of puncta centroids across rounds

rnd_idx=5;
filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch00'));
sample_img = load3DTif_uint16(filename_in);
imgSize = size(sample_img);

outputImg = makeDebugImageOfPoints(squeeze(pos(:,rnd_idx,:))',imgSize);
save3DTif_uint16(outputImg,fullfile(params.punctaSubvolumeDir,sprintf('transcript_pos_rnd%i.tif',rnd_idx)))

%% Get the cross talk matrix 

num_puncta = size(puncta_set,6);
%Creating a matrix of cross-talk in terms of pixel counts
%it might be then necessary to do a cross-talk of zscores
cross_talk_observed = zeros(4,4,num_puncta);
for puncta_idx = 1:num_puncta
    for rnd_idx = 1:params.NUM_ROUNDS
       %Get the base
       basecall = final_transcripts(puncta_idx,rnd_idx);
       
       for c_idx = 1:params.NUM_CHANNELS
          puncta = puncta_set(:,:,:,rnd_idx,c_idx,puncta_idx);
          puncta = puncta(:);
          puncta_masked = puncta(puncta>0);
          cross_talk_observed(basecall,c_idx,puncta_idx) = mean(puncta_masked);
       end
    end
end

cross_talk_observed_total = zeros(4,4);
for row_idx = 1:4
    for col_idx= 1:4
        vector_across_all = squeeze(cross_talk_observed(row_idx,col_idx,:));
        cross_talk_observed_total(row_idx,col_idx) = mean(vector_across_all(vector_across_all>0));
    end
    %Then normalize each row
    cross_talk_observed_total(row_idx,:) = cross_talk_observed_total(row_idx,:) / ...
        sum(cross_talk_observed_total(row_idx,:));
end

figure; imagesc(cross_talk_observed_total);

%% Get the cross talk matrix (in z-score)

puncta_set_zscore = zeros(size(puncta_set));

for c = params.COLOR_VEC
    chan_col = reshape(puncta_set(:,:,:,:,c,:),[],1);
    col_normed = zscore(chan_col);
    puncta_set_zscore(:,:,:,:,c,:) = reshape(col_normed,size(squeeze(puncta_set(:,:,:,:,c,:))));
end


num_puncta = size(puncta_set,6);
%Creating a matrix of cross-talk in terms of pixel counts
%it might be then necessary to do a cross-talk of zscores
cross_talk_observed = zeros(4,4,num_puncta);
for puncta_idx = 1:num_puncta
    for rnd_idx = 1:params.NUM_ROUNDS
       %Get the base
       basecall = final_transcripts(puncta_idx,rnd_idx);
       
       for c_idx = 1:params.NUM_CHANNELS
          puncta = puncta_set_zscore(:,:,:,rnd_idx,c_idx,puncta_idx);
          puncta = puncta(:);
          puncta_masked = puncta(puncta>0);
          cross_talk_observed(basecall,c_idx,puncta_idx) = mean(puncta_masked);
       end
    end
end

cross_talk_observed_total = zeros(4,4);
for row_idx = 1:4
    for col_idx= 1:4
        vector_across_all = squeeze(cross_talk_observed(row_idx,col_idx,:));
        cross_talk_observed_total(row_idx,col_idx) = mean(vector_across_all(vector_across_all>0));
    end
    %Then normalize each row
    cross_talk_observed_total(row_idx,:) = cross_talk_observed_total(row_idx,:) / ...
        sum(cross_talk_observed_total(row_idx,:));
end

figure; imagesc(cross_talk_observed_total);
colorbar;
title('Cross-talk matrix of zscore')