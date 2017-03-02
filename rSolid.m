load(fullfile(params.punctaSubvolumeDir, 'transcriptsv3.mat'));
load('/Users/Goody/Neuro/ExSeq/rajlab/splintr1/rois_votednonnormed.mat');

loadParameters;
%TODO: the punctaset collection contains that are too close to the boundary
%so we filter those out in the extract_transcripts file using the 
%good_puncta_indices vector. The 'bad' indices of the puncta_set simply
%contain zeros. So we need to first get the transcripts and puncta_set 
%vectors to agree
puncta_set = puncta_set(:,:,:,:,:,good_puncta_indices);
Y = Y(good_puncta_indices);
X = X(good_puncta_indices);
Z = Z(good_puncta_indices);


%The puncta_set vector is of dimensions:
%(PUNCTA_SIZE,PUNCTA_SIZE,PUNCTA_SIZE,NUM_ROUNDS,NUM_CHANNELS,num_puncta)

%Let's get all the pixels per channel per round

% NUM_ROUNDS = 3;
% NUM_COLORS = 4;
% DISTANCE_FROM_CENTER = 2.5;
puncta_vector = zeros(size(puncta_set,6),params.NUM_ROUNDS,params.NUM_COLORS);
for z = 1:size(puncta_vector,1)
    for exp_idx = 1:params.NUM_ROUNDS
        
        punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,z));
        
        max_intensity = max(max(max(max(punctaset_perround))))+1;
        min_intensity = min(min(min(min(punctaset_perround))));
        
        [max_chan, confidence,scores] = chooseChannelMod(punctaset_perround,params.COLOR_VEC,params.DISTANCE_FROM_CENTER);
        puncta_vector(z,exp_idx,:) = scores;
    end
    
    if(mod(z,100)==0)
        fprintf('Processing puncta #%i out of %i \n',z, size(puncta_vector,1));
    end
    
end


%%

HIST_COUNTS = 100;
transcripts = zeros(size(puncta_vector,1),params.NUM_ROUNDS);
transcripts_confidence = zeros(size(puncta_vector,1),params.NUM_ROUNDS);

% puncta_vector_divmean = puncta_vector(:,r,:) ./ ...
%     repmat(mean(squeeze(puncta_vector(:,r,:)),2),size(puncta_vector,1),1);
for r = 1:params.NUM_ROUNDS
 
    f0 = [];
    f1 = [];
    pc = zeros(1,4);

    mean_vec = mean(squeeze(puncta_vector(:,r,:)),1);
    %temp hack for empty channel of all zeros.
    mean_vec(3) = 1;
    puncta_vector_divmean = squeeze(puncta_vector(:,r,:)) ./ ...
    repmat(mean_vec,size(puncta_vector,1),1);

    for p = 1:size(puncta_vector,1)
        
%         t = squeeze(puncta_vector_divmean(p,r,:));
        t = squeeze(puncta_vector_divmean(p,:));
        [vals, I] = sort(t,'descend');
        transcripts(p,r) = I(1);
        transcripts_confidence(p,r) = vals(1)/vals(2);
        %Add to the vectors that we'll create the signal/bg distributions
        f1 = [f1; vals(1)];
        f0 = [f0; vals(2:3)'];
        
        %Keep track of how many times each channel is the brightest
        pc(I(1)) = pc(I(1))+1;
    end
    
%     figure; histogram(f0); title(sprintf('Background Round %i'r));
%     figure; histogram(f1); title(sprintf('Signal Round %i',r));
    
    pc = pc/sum(pc)
    
    %takign the log2 of the f0 and f1 distributions
    f0 = log2(f0);
    f1 = log2(f1);
    
    %When creating the two different distributions, f0 and f1, we have to 
    %make sure that we're binning them in the same way. So we'll first 
    %concatenate the two to create the histogram bins, then use those
    %binedges to produce f1dist and f0dist
    [ftotdist,binedges] = histcounts([f0;f1],HIST_COUNTS);
    [f1dist,binedges] = histcounts(f1,binedges);
    [f0dist,binedges] = histcounts(f0,binedges);
    
%     figure; 
%     b = bar(binedges(1:HIST_COUNTS),ftotdist,'r');b.FaceAlpha = 0.3; hold on;
%     b = bar(binedges(1:HIST_COUNTS),f1dist,'g'); b.FaceAlpha = 0.3;
%     b = bar(binedges(1:HIST_COUNTS),f0dist,'b');b.FaceAlpha = 0.3;
%     legend('Total distribution','F1','F0')
%     title(sprintf('Round %i',r));
%     
%     pause; 
%     
%     %Make the reference distributions per color (just in one round)
%     f1_ref = (1-pc(1))*f0dist + pc(1)*f1dist;
%     f2_ref = (1-pc(2))*f0dist + pc(2)*f1dist;
%     f4_ref = (1-pc(4))*f0dist + pc(4)*f1dist;
    
    %Normalize the data into a Nx3 vector
%     normed_puncta_vals(:,1) = quantilenorm_toref([squeeze(puncta_vector(:,r,1)),f1_ref]);
%     normed_puncta_vals(:,2) = quantilenorm_toref([squeeze(puncta_vector(:,r,2)),f2_ref]);
%     normed_puncta_vals(:,3) = squeeze(puncta_vector(:,r,3)); %All zeros in splintr case
%     normed_puncta_vals(:,4) = quantilenorm_toref([squeeze(puncta_vector(:,r,4)),f4_ref);
%     
    
end
