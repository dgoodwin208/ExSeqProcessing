% Explore permutations of likely misreads
%Get prob_transcripts for this file
loadParameters;

if ~exist('X','var')
    load(fullfile(params.punctaSubvolumeDir,'puncta_rois.mat'),'X','Y','Z');
end

if ~exist('maxProj','var')
    img = load3DTif(fullfile(params.rajlabDirectory,'alexa005.tiff'));
    maxProj = max(img,[],3); clear img;
end

%%
%prob_transcripts is num_puncta x rounds x channels
%X, Y, Z are the positions (in puncta_roi.mat)


%First, get a sense of confidence distributions across rounds
total_confidences = zeros(1,size(transcripts,1)*params.NUM_ROUNDS);
for puncta_idx = 1:size(transcripts,1)
    
    puncta_panel = squeeze(transcripts_intensities(puncta_idx,:,:));
    
    [v,I] = sort(puncta_panel,2,'descend');
    
    %Convert to probability from the log
    total_confidences((puncta_idx-1)*params.NUM_ROUNDS+1:(puncta_idx)*params.NUM_ROUNDS) ...
        = v(:,1)./v(:,2);
    
end

%Correct for zeros, just turn them to 1
% total_confidences(total_confidences==0) =1;
percentiles = prctile(total_confidences,[1:100]);
figure;
plot(percentiles)
ylabel('Confidence score')
title('Confidence score (for probability lower is better)');
xlabel('Percentile')
%%

%If the confidence is lower than this, mark it for toggling
CONFIDENCE_THRESHOLD = 20; %What is the percentile cutoff?
MAXIMUM_TOGGLES = 0;

% total_output =
complete_options = cell(size(transcripts,1),1);
% explanatory_strings = {};
for puncta_idx = 1:size(transcripts,1)
    
    %Get all the 2D information of scores for a puncta
    
    puncta_panel = squeeze(transcripts_intensities(puncta_idx,:,:));
    
    %Get the sorted values and indices
    [v,I] = sort(puncta_panel,2,'descend');
    
    round_confidences = exp(v(:,1))./exp(v(:,2));
    
    %Get the percentile of the confidence per round of sequencing
    prc = zeros(params.NUM_ROUNDS,1);
    for rnd_idx = 1:params.NUM_ROUNDS
        %Get the percentile that the confidence lays in
        
        %NOTE: I'm still not entirely sure when we get a NaN vs Inf
        if round_confidences(rnd_idx)>percentiles(100)
            prc(rnd_idx) = 100;
        elseif isnan(round_confidences(rnd_idx))
            prc(rnd_idx) = 0;
        else
            prc(rnd_idx) = find(round_confidences(rnd_idx)-percentiles<0,1);
        end
    end
    
    
    %Get all the rounds that need permutation
    toggle_indices = [];
    %For the same 1:n in toggle_indices, put the other bases
    %Ordered by likelihood
    possible_bases = {};
    ctr = 1;
    %Get the sorted values and indices
    [vperc,Iperc] = sort(prc,'ascend');
    for sorted_percentile_index = 1:MAXIMUM_TOGGLES
        rnd_idx = Iperc(sorted_percentile_index);
        %If the percentile confidence is high (ie, close to 1), then
        if prc(rnd_idx) <CONFIDENCE_THRESHOLD
            toggle_indices(ctr) = rnd_idx;
            %Take the top two choices from the original sorting
            possible_bases{ctr} = [I(rnd_idx,1),I(rnd_idx,2)];
            ctr = ctr+1;
        else
            break
        end
    end
    
    %If there were none to toggle, simply take the best transcript
    %probability
    if ctr==1
        complete_options{puncta_idx} = {I(:,1)};
        continue
    end
    
    fprintf('Puncta idx %i, Number of toggles: %i\n',puncta_idx, ctr-1);
    %Generate all permutation from those indices
    complete_options{puncta_idx} = generateToggles(I(:,1),toggle_indices,possible_bases);
    
    %Make a demo image that shows the transcript originally and all the
    %perms
    %     output_img = zeros(params.NUM_ROUNDS,1+length(complete_options{filtered_puncta_idx}));
    %     output_img(:,1) = I(:,1);
    %     for x=2:size(output_img,2)
    %         output_img(:,x) = complete_options{filtered_puncta_idx}{x-1};
    %     end
    %     figure(3)
    %     imagesc(output_img); colormap gray
    %     pause
    
end


%%

% load(fullfile(params.punctaSubvolumeDir,'puncta_rois.mat'));
close all;
% Map the colors of the experiment to what bowtie expects
%  blue=0, green=1, yellow=2 and red=3
channel_mapping = [0,1,3,2];

%What round of sequencing is N-1P?
ALIGNMENT_START_IDX = 4;
ALIGNMENT_STOP_IDX = 25;
KEEP_CODE = [1;4;2];
DO_PLOT = false; hasInitGif = 0; filename='puncta_transcripts_exseq.gif';
clear data
output_ctr = 1;


for t_idx = 1:size(transcripts,1)
    
    if(mod(t_idx,100)==0)
        fprintf('Writing Puncta idx %i\n',t_idx)
    end
    
    %Get all pertubations for that transcript:
    transcript_pertubations = complete_options{t_idx};
    num_array = []; t_option_ctr = 1;
    for t_option_idx = 1:length(transcript_pertubations)
        
        candidate_transcript = transcript_pertubations{t_option_idx};
        
%         if all(candidate_transcript(1:3)==KEEP_CODE)
            num_array(t_option_ctr,:) = candidate_transcript;
            t_option_ctr = t_option_ctr+1;
%         else
            %fprintf('Discarding option that had incorrect Primer reads:%i%i%i \n',candidate_transcript(1:3));
%         end
    end
    
    if sum(size(num_array)==0)
        fprintf('Discarding option that no correct Primer reads \n');
        continue
    end
    %Now, for each pertubation of that transcript, get the string
    for t_option_idx = 1:size(num_array,1)
        all_sequences = {}; %For deduplication
        dedupe_seq_ctr = 1;
        t_out_num = num_array(t_option_idx,:);
        
        string_ctr = 1;
        
        t_out_string = '';
        for base_idx = 1:params.NUM_ROUNDS
            
            t_out_string(string_ctr) = num2str(channel_mapping(t_out_num(base_idx)));
            string_ctr = string_ctr+1;
        end
        
        if t_option_idx==1
            top_guess_string = t_out_string;
        end
        
        if ~isempty(strmatch(t_out_string(ALIGNMENT_START_IDX:ALIGNMENT_STOP_IDX),all_sequences))
            fprintf('Skipping a duplicate sequence\n');
        else
            data(output_ctr).Sequence = t_out_string(ALIGNMENT_START_IDX:ALIGNMENT_STOP_IDX);
            data(output_ctr).Header = sprintf('idx=%i,Y=%i,X=%i,Z=%i',t_idx,...
                Y(t_idx),X(t_idx),Z(t_idx));
            all_sequences{dedupe_seq_ctr} = t_out_string(ALIGNMENT_START_IDX:ALIGNMENT_STOP_IDX);
            dedupe_seq_ctr = dedupe_seq_ctr+1;
            output_ctr = output_ctr +1;
        end
    end
    
    %Print out strings to show the various options per string
    %The top choice is the first element in num_array
    %And we produce strings to show the variance around it
%     options_string = repmat('-',1,length(t_out_string));
%     indices_that_were_toggled = std(num_array(:,ALIGNMENT_START_IDX:ALIGNMENT_STOP_IDX),0,1)>0;
%     options_string(indices_that_were_toggled) = 'X';
%     fprintf('%s\n%s\n\n',top_guess_string,options_string);
    
    %PLOTTING FIGURE
    if(DO_PLOT)
        figure(1);
        clf('reset')
        ha = tight_subplot(params.NUM_ROUNDS,params.NUM_CHANNELS,zeros(params.NUM_ROUNDS,2)+.01);
        
        subplot_idx = 1;
        for exp_idx = 1:params.NUM_ROUNDS
            
            punctaset_perround = squeeze(puncta_set_filtered(:,:,:,exp_idx,:,t_idx));
            
            max_intensity = max(max(max(max(punctaset_perround))))+1;
            min_intensity = min(min(min(min(punctaset_perround))));
            values = zeros(4,1);
            for c_idx = 1:params.NUM_CHANNELS
                
                
                clims = [min_intensity,max_intensity];
                %Get the subplot index using the tight_subplot system
                axes(ha(subplot_idx));
                
                punctaVol = squeeze(punctaset_perround(:,:,:,c_idx));
                imagesc(max(punctaVol,[],3),clims);
                
                if c_idx==1
                    axis off;
                    text(-0.0,10.,[top_guess_string(exp_idx), options_string(exp_idx)],'rotation',90)
                    %                 ylabel(top_guess_string(exp_idx));
                    %                 axis tight;
                    
                else
                    axis off;
                end
                
                if exp_idx==1
                    title(sprintf('%i',channel_mapping(c_idx)));
                end
                colormap gray
                subplot_idx = subplot_idx+1;
            end
        end
        
        drawnow
        if hasInitGif==0
            pause
        end
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if hasInitGif==0
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
            hasInitGif = 1;
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
        
        
        %         figure(2);
        %
        %         imagesc(maxProj);
        %         hold on;
        %         plot(X(t_idx),Y(t_idx),'r.');
        %         hold off;
        %         axis off;
        %         title(top_guess_string + ' Max intensity for Round5' );
        %         pause
    end
    
end

disp('Done!')
%%
fprintf('Total transcripts variants %i for %i puncta\n',output_ctr-1,size(puncta_set,6));
fastafile_output = fullfile(params.punctaSubvolumeDir,sprintf('transcripts_total%i.csfasta',MAXIMUM_TOGGLES));
if exist(fastafile_output,'file')
    delete(fastafile_output);
end
fastawrite(fastafile_output,data)

% fastqfile_output = fullfile(params.punctaSubvolumeDir,'transcripts.csfastq');
% if exist(fastqfile_output,'file')
%     delete(fastqfile_output);
% end
% fastqwrite(fastqfile_output,data)


%% Just write the intensity transcripts

close all;
% Map the colors of the experiment to what bowtie expects
%  blue=0, green=1, yellow=2 and red=3
channel_mapping = [0,1,3,2];

%What round of sequencing is N-1P?
ALIGNMENT_START_IDX = 4;
KEEP_CODE = [1,4,2];
clear data
output_ctr = 1;


for t_idx = 1:size(transcripts_probfiltered,1)
    
    if(mod(t_idx,100)==0)
        fprintf('Writing Puncta idx %i\n',t_idx)
    end
    
    
    t_out_num = transcripts_probfiltered(t_idx,:);
    
    if ~all(t_out_num(1:3)==KEEP_CODE)
        fprintf('Skipping puncta %i because primer read didnt match: %i%i%i\n',t_idx,t_out_num(1:3));
        continue
    end
            
    string_ctr = 1;
    
    t_out_string = '';
    for base_idx = 1:params.NUM_ROUNDS
        t_out_string(string_ctr) = num2str(channel_mapping(t_out_num(base_idx)));
        string_ctr = string_ctr+1;
    end
    

       
    data(output_ctr).Sequence = t_out_string(ALIGNMENT_START_IDX:params.NUM_ROUNDS);
    data(output_ctr).Header = sprintf('idx=%i,Y=%i,X=%i,Z=%i',t_idx,...
        Y(t_idx),X(t_idx),Z(t_idx));
    
    output_ctr = output_ctr +1;
    
    
    %Print out strings to show the various options per string
    %The top choice is the first element in num_array
    %And we produce strings to show the variance around it
    
    fprintf('%s\n',t_out_string);

end

disp('Done!')
fprintf('Total transcripts written %i for %i puncta\n',output_ctr-1,size(puncta_set_filtered,6));
fastafile_output = fullfile(params.punctaSubvolumeDir,'transcripts_intensityonly.csfasta');
if exist(fastafile_output,'file')
    delete(fastafile_output);
end
fastawrite(fastafile_output,data)