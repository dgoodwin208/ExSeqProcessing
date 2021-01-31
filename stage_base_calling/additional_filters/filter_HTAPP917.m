
%In the case of HTAPP917, we had an issue with extremely bright artifacts
%being present in the bottom of the gel, present only in the first two
%channels, and increasing in intensity throughout the rounds, peaking at
%round 5
% This function will identify the bad puncta across the rounds, then remove
% them for later for processing


%If a puncta has a round with the first two channels over this ERRONEOUS
%BRIGHTNESS, remove that puncta.
ERRONEOUS_BRIGHTNESS = 1000; 
CHANNELS_TO_MONITOR = 1:2;

%We'll use the mean value to determine the excessive brightness
puncta_meanvals = zeros(N, readlength,4); 

N = size(puncta_set_cell{1},1);
for rnd_idx = 1:readlength
    for p_idx = 1:N
        for c_idx = 1:4
            puncta_meanvals(p_idx,rnd_idx,c_idx) = mean(puncta_set_cell{rnd_idx}{p_idx,c_idx});
        end
    end    
end

% To confirm that the cropped fov worked out well, let's make sure there
% are no puncta with more than 2 missing rounds
%%
puncta_bad = zeros(N,readlength);
for rnd_idx = 1:readlength
    for p_idx = 1:N
        colormax_per_round = squeeze(puncta_meanvals(p_idx,rnd_idx,CHANNELS_TO_MONITOR));
        puncta_bad(p_idx,rnd_idx) = all(colormax_per_round>ERRONEOUS_BRIGHTNESS);
    end
end

puncta_discard = any(puncta_bad,2);
puncta_keep = 1:N; puncta_keep(puncta_discard) = [];
fprintf('HTAPP_917 filter: Discarding %i puncta, keeping %i\n',sum(puncta_discard),length(puncta_keep));


%% Get the spatial position of these removals
puncta_voxels = puncta_indices_cell{1};
pos = zeros(N,3);
for p = 1:N
    [x,y,z] = ind2sub(IMG_SIZE,puncta_voxels{p});
    pos(p,:) = mean([x,y,z],1); 
end

figure; %('Visible','off');
plot3(pos(puncta_keep,1),pos(puncta_keep,2),pos(puncta_keep,3),'b.');
hold on;
plot3(pos(puncta_discard,1),pos(puncta_discard,2),pos(puncta_discard,3),'ro');
legend('Keep','Removed');
title(sprintf('%i puncta are over the intensity threshold %i in two channels in at least one round',sum(puncta_discard),ERRONEOUS_BRIGHTNESS));

save_type = 'jpg';
figfilename = fullfile(params.reportingDir,...
    sprintf('%s_%s_3Dplot.%s',...
    params.FILE_BASENAME,...
    'base-calling-filtering',...
    save_type));
saveas(gcf,figfilename,save_type)
        

%% 
figure;
bar(1:readlength,sum(puncta_bad,1))
title(sprintf('%i puncta are over the intensity threshold %i in two channels in at least one round',sum(puncta_discard),ERRONEOUS_BRIGHTNESS));
ylabel('Number of erroneous puncta due to artifacts by round')
xlabel('Sequencing Round');

save_type = 'jpg';
figfilename = fullfile(params.reportingDir,...
    sprintf('%s_%s_artifactsByRound.%s',...
    params.FILE_BASENAME,...
    'base-calling-filtering',...
    save_type));
saveas(gcf,figfilename,save_type)

%% Save the output
% TODO: The output of punctafeiner is NUMROUNDS of identical indices. This is
% redundant for now, and to fix this properly we should go back into the
% punctafeinder code.
puncta_indices_cell_filtered = puncta_indices_cell;
puncta_set_cell_filtered = puncta_set_cell;
for rnd_idx = 1:params.NUM_ROUNDS
    puncta_indices_cell_filtered{rnd_idx} = puncta_voxels(puncta_keep,:);
    pixels_per_rnd = cell(length(puncta_keep),params.NUM_CHANNELS);
    for p_idx = 1:length(puncta_keep)
        for c_idx = 1:4
            pixels_per_rnd{p_idx,c_idx}  = puncta_set_cell{rnd_idx}{puncta_keep(p_idx),c_idx};
        end
    end
    puncta_set_cell_filtered{rnd_idx} = pixels_per_rnd;
end

