
successes = 0;
for t_idx = 1:size(transcripts,1)
    if all(transcripts(t_idx,2:3) == [3 2])
        successes = successes +1;
    end
end

successes

%% Score it
for base_idx = 1:params.NUM_CHANNELS
    perc_base(:,base_idx) = sum(transcripts==base_idx,1)/size(transcripts,1);
end
figure;
% Chan 1 = Blue
% Chan 2 = Green
% Chan 3 = Magenta
% Chan 4 = Red

plot(perc_base(:,1)*100,'b','LineWidth',2); hold on;
plot(perc_base(:,2)*100,'g','LineWidth',2)
plot(perc_base(:,3)*100,'m','LineWidth',2)
plot(perc_base(:,4)*100,'r','LineWidth',2); hold off;
legend('Chan1 - FITC','Chan2 - CY3', 'Chan3 - Texas Red', 'Chan4 - Cy5');
title(sprintf('Percentage of each base across rounds for %i puncta',size(transcripts,1)));


%% Quick and dirty scoring of transcripts 

hamming_scores = zeros(size(transcripts,1),1);
% par_factor = 5;

%Size 17 because we score from round 4-20
round_mask = ones(1,17);
% round_mask(11) = 0; %ignore round 14
% round_mask(2) = 0; %ignore round 5
round_mask = logical(round_mask);


for p_idx = 1:size(transcripts,1)
    
%     transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = transcripts(p_idx,4:end);
%     img_transcript = [transcripts(p_idx,:) 0];
    %Sanity check: randomize the img_transcript
 
    %Search for a perfect match in the ground truth codes
    hits = (groundtruth_codes(:,round_mask)==img_transcript(round_mask));
    

    %Calculate the hamming distance 
    scores = length(img_transcript)- sum(hits,2) - sum(~round_mask);
%     [values, indices] = sort(scores,'ascend');    
%     hamming_scores(p_idx) = values(1);
    hamming_scores(p_idx) = min(scores);
    
    if mod(p_idx,500) ==0
        fprintf('%i/%i matched\n',p_idx,size(transcripts,1));
    end
end
 
hamming_clipped = hamming_scores(1:p_idx-1);
figure;
histogram(hamming_clipped,length(unique(hamming_clipped)))
%%
load('/Users/Goody/Neuro/ExSeq/exseq20170524/exseqautoframe7_puncta_allexp.mat');
load('/Users/Goody/Neuro/ExSeq/exseq20170524/exseqautoframe7_punctamap.mat');
puncta{REF_IDX} = removeRedundantPuncta(puncta{REF_IDX});
%%
%First go through all the rounds to get the indices of REF_IDX puncta that
%are present in all 19 rounds (in this case the first primer round
%registration failed
REF_IDX = 5;

votes_refidx = zeros(size(puncta{REF_IDX},1),20);
for rnd_idx = 1:params.NUM_ROUNDS
    
    punctamap = punctamap_master{rnd_idx};
    
    for nn_idx = 1:length(punctamap)
        %votes refidx is indexed the same as the reference round
        votes_refidx(punctamap{nn_idx}.index_ref,rnd_idx)=1;
    end
    
end

%make a plot to show the number of trackable puncta through n rounds
total_counts_per_ref = sum(votes_refidx,2);

for idx = 1:19
    track_counts(idx) = sum(total_counts_per_ref>=idx);
end

figure;
subplot(1,2,1)
plot(1:19,track_counts);
title('Number of rounds a puncta can be tracked through given cutoff distance of 10');

subplot(1,2,2)
bar(1:20,sum(votes_refidx,1));
title('Number of puncta per round within distance of 10 from Reference round');



%% cutoffs and interpolate

MIN_NEIGHBOR_AGREEMENT = 17;

REF_INDICES_TO_KEEP = find(total_counts_per_ref>=MIN_NEIGHBOR_AGREEMENT);

%Create an array of all the centroid locations across rounds for a single
%puncta
centroid_collections = zeros(length(REF_INDICES_TO_KEEP),params.NUM_ROUNDS,3);
puncta_roll_call = zeros(length(REF_INDICES_TO_KEEP),params.NUM_ROUNDS);
%For each puncta that pass
ctr_progress = 0;
for ref_puncta_idx = squeeze(REF_INDICES_TO_KEEP)'
    
    rounds_that_need_averaging = ones(params.NUM_ROUNDS,1);
    rounds_that_need_averaging(REF_IDX) = 0;
    
    %This needs to be double checked:
%     centroid_collections(ref_puncta_idx,REF_IDX,:) = puncta{REF_IDX}(ref_puncta_idx,:);
    
    noted_refLoc = 0;
    for rnd_idx = 1:params.NUM_ROUNDS
        
        if rnd_idx==REF_IDX
            continue;
        end
        
        punctamap = punctamap_master{rnd_idx};
        
        %find the idx of the punctamap object that matches
        puncta_obj = struct;

        for nn_idx = 1:length(punctamap)
            if punctamap{nn_idx}.index_ref == ref_puncta_idx
                puncta_obj = punctamap{nn_idx};
                rounds_that_need_averaging(rnd_idx)=0;
                break;
            end
        end
        
        if rounds_that_need_averaging(rnd_idx)
%             fprintf('Puncta %i not present in round %i\n',ref_puncta_idx, rnd_idx);
            continue;
        end
        
        centroid_collections(ref_puncta_idx,rnd_idx,:) = puncta_obj.pos_moving;
        
        if ~noted_refLoc
            centroid_collections(ref_puncta_idx,REF_IDX,:) = puncta_obj.pos_ref;
            noted_refLoc=1;
        end
    end
    %For the rounds that did not have a puncta match to the ref round
    %take the average of the rounds that did have a match
    centroid_collections(ref_puncta_idx,logical(rounds_that_need_averaging),:) = ...
            repmat(...
            mean(squeeze(centroid_collections(ref_puncta_idx,~logical(rounds_that_need_averaging),:)),1), ...
            sum(rounds_that_need_averaging),1);
        
    puncta_roll_call(ref_puncta_idx,:) = rounds_that_need_averaging;
    
    ctr_progress = ctr_progress+1;
    
    if mod(ctr_progress,100)==0
       fprintf('Processed %i of %i puncta\n',ctr_progress, length(REF_INDICES_TO_KEEP));
    end
    
end

save(fullfile(params.punctaSubvolumeDir,sprintf('%s_centroid_collections.mat',params.FILE_BASENAME)),'centroid_collections','-v7.3','puncta_roll_call');

%How do we handle the case of puncta that aren't found? We take the average


%%
% figure;
% scatter3(Y(duplicates,1),Y(duplicates,2),Y(duplicates,3),'r'); hold on;
% scatter3(X(IDX(duplicates),1),X(IDX(duplicates),2),X(IDX(duplicates),3),'b');
%
% duplicate_linear_indices = 1:size(Y,1);
% duplicate_linear_indices = duplicate_linear_indices(duplicates);
%
% for idx = duplicate_linear_indices
%     lines = [ ...
%         [Y(idx,1);X(IDX(idx),1)] ...
%         [Y(idx,2);X(IDX(idx),2)] ...
%         [Y(idx,3);X(IDX(idx),3)] ];
%
%     rgb = [0 0 0];
%     if lines(1,1) > lines(2,1)
%         rgb(1) = .7;
%     end
%     if lines(1,2) > lines(2,2)
%         rgb(2) = .7;
%     end
%     if lines(1,3) > lines(2,3)
%         rgb(3) = .7;
%     end
%     plot3(lines(:,1),lines(:,2),lines(:,3),'color',rgb);
% end