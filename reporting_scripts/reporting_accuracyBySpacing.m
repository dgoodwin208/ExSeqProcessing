loadParameters;

filename_centroidsORIG = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',params.FILE_BASENAME));
load(filename_centroidsORIG);

%% Load the number of puncta in each round and get an average

num_puncta_per_round = zeros(params.NUM_ROUNDS,1);
for rnd_idx = 1:params.NUM_ROUNDS
    num_puncta_per_round(rnd_idx) = size(puncta_centroids{rnd_idx},1);
end

clear puncta_centroids puncta_voxels puncta_baseguess;
%% For the final puncta, score each index with it's distance to nearests neightbor
% filename_centroidsMOD = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels_demerged.mat',params.FILE_BASENAME));
% load(filename_centroidsMOD);
spacings = cell2mat(cellfun(@(x) x.nn_distance, transcript_objects,'UniformOutput',0));
figure;
histogram(spacings)  
title(sprintf('Histogram of distance to nearest neighbor puncta, N=%i',length(spacings)));
xlabel('Distance (pixels)');
ylabel('Count');

%% Look at the histograms by the hamming distance
scoresAndDistances = cell2mat(cellfun(@(x) [x.hamming_score x.nn_distance], transcript_objects,'UniformOutput',0));

figure;
subplot(max(scoresAndDistances(:,1))+1,1,1);

for specific_hamming_distance = 0:max(scoresAndDistances(:,1))
      
    subplot(max(scoresAndDistances(:,1))+1,1,specific_hamming_distance+1);
    
    indices = find(scoresAndDistances(:,1)==specific_hamming_distance);
    histogram(scoresAndDistances(indices,2))
    title(sprintf('Histogram of nearest-neighbor distances for Hamming Score = %i, N=%i',specific_hamming_distance,length(indices)));
    xlabel('Distance (pixels)');
end

%% Consolidate the results 

%load(fullfile(directory_to_process,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)));

possible_spacings = unique(spacings);

puncta_by_spacing_good = zeros(length(possible_spacings),1);
puncta_by_spacing_bad = zeros(length(possible_spacings),1);
% puncta_indices = cell(length(possible_spacings),1);

for puncta_idx = 1:length(transcript_objects)
    isGoodPuncta = transcript_objects{puncta_idx}.hamming_score<=1;
    
    %Get the index of the vector given the distance
    bucket_idx = find(possible_spacings==spacings(puncta_idx));
    if isGoodPuncta
        puncta_by_spacing_good(bucket_idx) = puncta_by_spacing_good(bucket_idx)+1;
    else
        puncta_by_spacing_bad(bucket_idx) = puncta_by_spacing_bad(bucket_idx)+1;
    end
end

for s_idx = 1:length(possible_spacings)
   total_per_spacing = puncta_by_spacing_good(s_idx)+ puncta_by_spacing_bad(s_idx);
   fprintf('%i\t%i\t%i\n',possible_spacings(s_idx),total_per_spacing,puncta_by_spacing_good(s_idx));
end