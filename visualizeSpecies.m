 CStr = textread('../filtered_transcripts.txt', '%s', 'delimiter', '\n');
 
 
 
 %Load a sample file
 data = load3DTif('/Users/Goody/Neuro/ExSeq/culture/input/sa0916dncv_round1_summed.tif');
 
 gene_idx = 0;
 gene_names = {};
 gene_space = {};
 transcript_locs= []; trans_idx =1;
 %% Load all 
 for row_idx = 1:length(CStr)
    entry = CStr{row_idx};
    elts = split(entry);
    
    %If it's a new species
    if strcmp(elts{1},'Colors:') 
        try
            gene_idx = gene_idx +1;
            gene_names{gene_idx}= elts{6};
            
            transcript_locs = []; trans_idx =1;
            
        catch
            disp('Skipping bad gene');
            continue
        end
        
    elseif strcmp(elts{1},'Location:')  %Parse the location
        y = str2double(elts{2});
        x = str2double(elts{3});
        z = str2double(elts{4});
        transcript_locs(trans_idx,:) = [y,x,z];
        trans_idx = trans_idx + 1;
        
        %insert it into the cell array for later visualization
        gene_space{gene_idx} = transcript_locs;
        
    end
 end
 
 
 %% Visualize the top 3
 colors = {'r.','g.','b.','y.','c.','m.','r*','g*','b*','y*','c*','m*'};
 figure(1);
 imagesc(max(data,[],3)); colormap gray
 hold on;
 
 for gene_idx=1:91
     
     transcript_locs = gene_space{gene_idx};
%      fprintf('Color: %s , Gene: %s\n',colors{gene_idx},gene_names{gene_idx});
     for trans_idx = 1:size(transcript_locs,1)
        plot(transcript_locs(trans_idx,2),transcript_locs(trans_idx,1),colors{gene_idx});
     end
 end
 