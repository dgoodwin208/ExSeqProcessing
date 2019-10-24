

puncta_present = ones(N,readlength);
%Get the total number of voxels involved for the puncta
%used to initialize data_cols
total_voxels = 0;
for n = 1:N
    total_voxels = total_voxels + length(puncta_set_cell_filtered{1}{n});
end

puncta_set_normalized = puncta_set_cell_filtered;

%Create a clims object for each round
%This will be later used for calculating the histogram settings for
%visualizing the puncta gridplots
clims_perround = zeros(4,2,readlength);

%We'll also do some new insitu_transcript base calling while we're at it
insitu_transcripts = zeros(N,readlength);
%Each image is normalized seperately
illumina_corrections = zeros(readlength,1);
for rnd_idx = 1:readlength
    
    %this vector keeps track of where we're placing each set of voxels per
    %color into a vector
    punctaindices_vecpos = zeros(N,1);
    pos_cur = 1;
    
    data_cols = zeros(total_voxels,4);
    for p_idx = 1:N
        voxels = puncta_set_cell_filtered{rnd_idx}{p_idx};
        n = length(voxels);
        
        %mark the next spot as we pack in the 1D vectors of the puncta
        pos_next = pos_cur+n-1;
        data_cols(pos_cur:pos_next,1) = puncta_set_cell_filtered{rnd_idx}{p_idx,1};
        data_cols(pos_cur:pos_next,2) = puncta_set_cell_filtered{rnd_idx}{p_idx,2};
        data_cols(pos_cur:pos_next,3) = puncta_set_cell_filtered{rnd_idx}{p_idx,3};
        data_cols(pos_cur:pos_next,4) = puncta_set_cell_filtered{rnd_idx}{p_idx,4};
        
        punctaindices_vecpos(p_idx) = pos_next;
        pos_cur = pos_next+1;
        
    end
    
    
    data_cols_norm = quantilenorm(data_cols);
    
    %
    %Unpack the normalized colors back into the original shape
    pos_cur = 1;
    for p_idx = 1:N
        puncta_set_normalized{rnd_idx}{p_idx,1} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),1);
        puncta_set_normalized{rnd_idx}{p_idx,2} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),2);
        puncta_set_normalized{rnd_idx}{p_idx,3} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),3);
        puncta_set_normalized{rnd_idx}{p_idx,4} = data_cols_norm(pos_cur:punctaindices_vecpos(p_idx),4);
        %Track the 1D index as we unpack out each puncta
        pos_cur = punctaindices_vecpos(p_idx)+1;
    end
    
    color_cutoffs = zeros(4,1);
    for c_idx = 1:4
        %We create histogram cutoffs from all the puncta in a particular
        %round. This is used for viewing later
        histval_bottom = prctile(data_cols_norm(:,c_idx),50);
        histval_top = prctile(data_cols_norm(:,c_idx),99);
        
        clims_perround(c_idx,1,rnd_idx) = histval_bottom;
        clims_perround(c_idx,2,rnd_idx) = histval_top;
        
        %Set a minimum
        color_cutoffs(c_idx) = 2*expfit(data_cols_norm(:,c_idx)); %prctile(data_cols_norm(:,c_idx),60);
    end
    
    %Filter out puncta that have a missing round, as determined by the
    %color cutoffs
    for p_idx = 1:N
        %A channel is present if the brightest pixel in the puncta are
        %brighter than the cutoff we just defined
        chan1_signal = prctile(puncta_set_normalized{rnd_idx}{p_idx,1},99);
        chan2_signal = prctile(puncta_set_normalized{rnd_idx}{p_idx,2},99);
        chan3_signal = prctile(puncta_set_normalized{rnd_idx}{p_idx,3},99);
        chan4_signal = prctile(puncta_set_normalized{rnd_idx}{p_idx,4},99);
        
        chan1_present = chan1_signal>color_cutoffs(1);
        chan2_present = chan2_signal>color_cutoffs(2);
        chan3_present = chan3_signal>color_cutoffs(3);
        chan4_present = chan4_signal>color_cutoffs(4);
        
        [~,winning_base] = max([chan1_signal,chan2_signal,chan3_signal,chan4_signal]);
        
        
        
        %Do a hardcoded illumina correction;
        %Red (chan1) can be bright without magenta (chan2), but
        %Magenta cannot be bright without red. So if chan2 is close to
        %chan1, call it 2.
        
        if abs(chan2_signal-chan1_signal)/chan2_signal<ILLUMINACORRECTIONFACTOR && winning_base==1
            winning_base=2;
            illumina_corrections(rnd_idx) = illumina_corrections(rnd_idx)+1;
        end
        
        insitu_transcripts(p_idx,rnd_idx) = winning_base;
        
        puncta_present(p_idx,rnd_idx) = chan1_present | chan2_present | ...
            chan3_present | chan4_present;
    end
    
    
    fprintf('Color cutoffs %s\n',mat2str(round(color_cutoffs)));
    fprintf('Completed Round %i\n',rnd_idx);
end
fprintf('Completed normalization!\n');

%A puncta is incomplete if it's never missing a round 
if params.ISILLUMINA
    puncta_complete = find(~any(puncta_present==false,2));
else
    puncta_complete = 1:N; %keep everything, don't filter
end
fprintf('Number of puncta before filtering missing bases: %i\n',N);
fprintf('Number of puncta before filtering missing bases: %i\n',length(puncta_complete));

N = length(puncta_complete);
puncta_set_normalized_filtered = puncta_set_normalized;
for rnd_idx = 1:readlength
    puncta_set_normalized_filtered{rnd_idx} = puncta_set_normalized{rnd_idx}(puncta_complete,:);
    puncta_indices_cell_filtered{rnd_idx} = puncta_indices_cell{rnd_idx}(puncta_complete);
end

insitu_transcripts_filtered = insitu_transcripts(puncta_complete,:);

puncta_voxels_filtered  = puncta_indices_cell_filtered{1};
puncta_centroids_filtered = zeros(N,3);
for p = 1:N
    [x,y,z] = ind2sub(IMG_SIZE,puncta_voxels_filtered{p});
    puncta_centroids_filtered(p,:) = mean([x,y,z],1);
    
end
