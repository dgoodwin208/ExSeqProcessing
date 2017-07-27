%Load the ROIS, calculate normalized intensities, then make transcript base
%calls

loadParameters;
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois_oversize.mat',params.FILE_BASENAME)));

%The puncta_set vector is of dimensions:
%(PUNCTA_SIZE,PUNCTA_SIZE,PUNCTA_SIZE,NUM_ROUNDS,NUM_CHANNELS,num_puncta)

PSIZE=size(puncta_set,1);
%% Vectorize the entire set of puncta into one column per channel

puncta_set_normed = zeros(size(puncta_set));
for c = params.COLOR_VEC
    chan_col(:,c) = reshape(puncta_set(:,:,:,:,c,:),[],1);
end
cols_normed = quantilenorm(chan_col);

for c = params.COLOR_VEC
    puncta_set_normed(:,:,:,:,c,:) = reshape(cols_normed(:,c),size(squeeze(puncta_set(:,:,:,:,c,:))));
end


%Rename puncta_set_normed to just puncta
puncta_set=puncta_set_normed;
%clearvars puncta_set_normed;

%%
%After the normalization, calculate fine-pixel tweaks per puncta

bad_puncta = [];
puncta_fix_cell = cell(size(puncta_set,6),1);

shifts_cell = cell(size(puncta_set,6),1);
for p_idx = 1:size(puncta_set,6)

    puncta = puncta_set(:,:,:,:,:,p_idx);
    puncta_backup = puncta;
    
    %The holder for all the shifts per puncta
    shifts_puncta = zeros(params.NUM_ROUNDS,3);
    
    %sum of channels (that have now been quantile normalized)
    puncta_roundref = sum(squeeze(puncta(:,:,:,params.REFERENCE_ROUND_PUNCTA,:)),4);
    offsetrange = [2,2,2];
    
    moving_exp_indices = 1:params.NUM_ROUNDS; 
    moving_exp_indices(params.REFERENCE_ROUND_PUNCTA) = [];

  
    
    for e_idx = moving_exp_indices
        %Get the sum of the colors for the moving channel
        puncta_roundmove = sum(squeeze(puncta(:,:,:,e_idx,:)),4);
        [~,shifts] = crossCorr3D(puncta_roundref,puncta_roundmove,offsetrange);
        if numel(shifts)>3
            %A maximum point wasn't found for this round, likely indicating
            %something weird with the round. Skip this whole puncta
            fprintf('Error in Round %i in Puncta %i\n', e_idx,p_idx);
            bad_puncta = [bad_puncta p_idx];
            puncta_fix_cell{p_idx} = puncta;
            break;
        end
        %If it's a good looking shift, save it
        
        shifts_puncta(e_idx,:) = shifts;

    end

    if ~ismember(p_idx,bad_puncta)
    for e_idx = moving_exp_indices        
        shifts= shifts_puncta(e_idx,:);
        for c_idx = 1:params.NUM_CHANNELS
            puncta(:,:,:,e_idx,c_idx) = ...
                imtranslate3D(squeeze(puncta(:,:,:,e_idx,c_idx)),shifts);
        end
    end    
    end
    %quick temp viewing
%     visualizeGridPlot(puncta,ones(params.NUM_ROUNDS,1),params,1)
%     visualizeGridPlot(puncta_backup,ones(params.NUM_ROUNDS,1),params,2)
%     [(1:20)', shifts_puncta]
% 
%     p_idx
%     drawnow
%     pause
shifts_cell{p_idx} = shifts_puncta;
   puncta_fix_cell{p_idx} = puncta;

   if mod(p_idx,1000)==0
       fprintf('3D crosscorr fix for puncta  %i/%i\n',p_idx,size(puncta_set,6));
   end
end
%%
%Bring it all back together
shifts_total = zeros(params.NUM_ROUNDS,3,size(puncta_set,6));
for p_idx = 1:size(puncta_set,6)
    puncta_set(:,:,:,:,:,p_idx) = puncta_fix_cell{p_idx};
    shifts_total(:,:,p_idx) = shifts_cell{p_idx};
end


%old code may have allowed bad_puncta to have duplicates
bad_puncta = unique(bad_puncta);

puncta_set(:,:,:,:,:,bad_puncta) = [];
X(bad_puncta) = [];
Y(bad_puncta) = [];
Z(bad_puncta) = [];
pos = [Y X Z];
save(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedrois_oversize.mat',params.FILE_BASENAME)),'puncta_set','pos','shifts_total','-v7.3');

%Hardcode the proper crop to get the subvolumes we're used to working with
puncta_set_normal= puncta_set(3:12,3:12,3:12,:,:,:);
puncta_set = puncta_set_normal; 
save(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedroisv12.mat',params.FILE_BASENAME)),'puncta_set','pos','shifts_total','-v7.3');

% Rows are sequencing rounds, columns are channels, press enter to go to
% next one
% transcripts = zeros(size(puncta_set,6),params.NUM_ROUNDS);
% transcripts_confidence = zeros(size(puncta_set,6),params.NUM_ROUNDS);
% pos = zeros(size(puncta_set,6),3);
% 
% cell_transcripts = cell(size(puncta_set,6),1);
% cell_transcripts_confidence = cell(size(puncta_set,6),1);
% cell_pos = cell(size(puncta_set,6),1);
% parfor puncta_idx = 1:size(puncta_set,6)
%     
%     answer_vector = zeros(params.NUM_ROUNDS,1);
%     confidence_vector = zeros(params.NUM_ROUNDS,1);
%     for exp_idx = 1:params.NUM_ROUNDS
%         
%         punctaset_perround = squeeze(puncta_set(:,:,:,exp_idx,:,puncta_idx));
%          
%         [max_chan, confidence] = chooseChannel(punctaset_perround,params.COLOR_VEC,params.DISTANCE_FROM_CENTER);
%         answer_vector(exp_idx) = max_chan;
%         confidence_vector(exp_idx) = confidence;
%     end
%     
%     %transcripts(puncta_idx,:) = answer_vector;
%     %transcripts_confidence(puncta_idx,:) = confidence_vector;
%     %pos(puncta_idx,:) = [Y(puncta_idx),X(puncta_idx),Z(puncta_idx)];
%     cell_transcripts{puncta_idx} = answer_vector;
%     cell_transcripts_confidence{puncta_idx} = confidence_vector;
%     cell_pos{puncta_idx} = [Y(puncta_idx),X(puncta_idx),Z(puncta_idx)];    
% 
%     if mod(puncta_idx,1000)==0
%         fprintf('Calling base puncta #%i out of %i \n',puncta_idx, size(puncta_set,6));
%     end
% end
% 
% for puncta_idx = 1:size(puncta_set,6)
%     transcripts(puncta_idx,:) = cell_transcripts{puncta_idx};
%     transcripts_confidence(puncta_idx,:) = cell_transcripts_confidence{puncta_idx};
%     pos(puncta_idx,:) = cell_pos{puncta_idx};
% end
% 
% 
% save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptsv12.mat',params.FILE_BASENAME)),'transcripts','transcripts_confidence','pos');
% save(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedroisv12.mat',params.FILE_BASENAME)),'puncta_set','pos','-v7.3');


disp('Completed normalizePuncta.m and saved the transcriptsv10 mat file in the transcripts folder');
