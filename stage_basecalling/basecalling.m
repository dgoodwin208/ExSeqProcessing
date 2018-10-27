

load(fullfile(params.basecallingResultsDir,sprintf('%s_puncta_pixels.mat',params.FILE_BASENAME)));

%% Because there is some Z dependence on intensity, we do naive basecalling
% per z slice

%Quick hack to make the rest of the code work:
puncta_set = puncta_set_mean;
puncta_set_bg = puncta_set_backgroundmedian;

ROUNDS_TO_CALL =  4:20; 
NUMCALLEDROUNDS = length(ROUNDS_TO_CALL);


%%  Convert all the data into zscores (very cheap base calling)
num_puncta = size(puncta_centroids,1);

%Pre-initialize the cell arrray and determine the basecalls

base_calls = zeros(num_puncta,params.NUM_ROUNDS);
base_calls_confidence = zeros(num_puncta,params.NUM_ROUNDS)-1;
base_calls_confidence_fast = zeros(num_puncta,params.NUM_ROUNDS)-1;
base_calls_secondplace = zeros(num_puncta,params.NUM_ROUNDS);

base_calls_rawpixel_intensity = zeros(num_puncta,params.NUM_ROUNDS,params.NUM_CHANNELS);
base_calls_normedpixel_intensity = zeros(num_puncta,params.NUM_ROUNDS,params.NUM_CHANNELS);


%%
clear dff fg bg chan_col;

for rnd_idx = 1:NUMCALLEDROUNDS
    
    actual_rnd_idx = ROUNDS_TO_CALL(rnd_idx);
    
    %fprintf('Looping through puncta from round %i\n',actual_rnd_idx);
    
    for c = params.COLOR_VEC
        dff = reshape(puncta_set(actual_rnd_idx,c,:),[],1);
        %dff = (reshape(puncta_set(actual_rnd_idx,c,:),[],1)...
        %    -reshape(puncta_set_bg(actual_rnd_idx,c,:),[],1))./...
        %    reshape(puncta_set_bg(actual_rnd_idx,c,:),[],1);
        fg(:,c) = reshape(puncta_set(actual_rnd_idx,c,:),[],1);    
        bg(:,c) = reshape(puncta_set_bg(actual_rnd_idx,c,:),[],1);   
        chan_col(:,c) = dff;
    end
    cols_normed = quantilenorm(chan_col);
    
    %how many puncta are less than the median?
    dimmer_than_surrounding = sum(fg,2)<sum(bg,2);
    %fprintf('Discarding %i puncta that are dimmer than the surroundings\n',...
    %    sum(dimmer_than_surrounding));

    %Sort each puncta (row) from loser to winner
    [vals, channel_order] = sort(cols_normed,2,'ascend');
    
    %Calculate a simple confidence first
    
    %Shift all normalized, ordered values, zerod to the next dimmest puncta
    %to ensure no weird negative values
    toptwo_channels = vals(:,3:4) - vals(:,1); 
    
    confidence_calc = toptwo_channels(:,2)./(toptwo_channels(:,2)+toptwo_channels(:,1));
    
    %Because we will have a few NaNs here, we'll set them to be
    %impossibly low confidence, which we can pickup later
    confidence_calc(isnan(confidence_calc)) = 0.1;
    base_calls_confidence_fast(:,actual_rnd_idx)=...
        min(-10*log10(1-confidence_calc),40);
    
    
    base_calls(:,actual_rnd_idx) = channel_order(:,4);
    base_calls_secondplace(:,actual_rnd_idx) = channel_order(:,3);
    
    
    %use the baseguess to get the absolute brightness of the puncta
    base_calls_rawpixel_intensity(:,actual_rnd_idx,:) = ...
        squeeze(puncta_set(actual_rnd_idx,:,:))';

    base_calls_normedpixel_intensity(:,actual_rnd_idx,:) = ...
        cols_normed;
        
    
end %end looping over rounds

%were there any NaNs in the computing? if so toss the read
bad_reads = find(any(base_calls_confidence_fast(:,ROUNDS_TO_CALL)<3,2));
fprintf('Have to discard %i reads because of an NaN in confidence\n',length(bad_reads));
base_calls(bad_reads,:) = [];
base_calls_secondplace(bad_reads,:) = [];
base_calls_confidence(bad_reads,:) = [];
base_calls_confidence_fast(bad_reads,:) = [];
base_calls_rawpixel_intensity(bad_reads,:,:) = [];
base_calls_normedpixel_intensity(bad_reads,:,:) = [];
puncta_centroids(bad_reads,:) = [];
puncta_voxels(bad_reads) = [];
[unique_transcipts,~,~] = unique(base_calls,'rows');
fprintf('Found %i transcripts, %i of which are unique\n', ...
    size(base_calls,1),size(unique_transcipts,1));


insitu_transcripts = base_calls(:,ROUNDS_TO_CALL);
insitu_transcripts_2ndplace = base_calls_secondplace(:,ROUNDS_TO_CALL);
insitu_transcripts_confidence = base_calls_confidence_fast(:,ROUNDS_TO_CALL);
insitu_transcripts_confidence_fast = insitu_transcripts_confidence;
save(fullfile(params.basecallingResultsDir,sprintf('%s_basecalls_meanpuncta.mat',params.FILE_BASENAME)),...
    'insitu_transcripts','insitu_transcripts_2ndplace','insitu_transcripts_confidence','insitu_transcripts_confidence_fast',...
    'base_calls_rawpixel_intensity','base_calls_normedpixel_intensity',...
    'puncta_centroids','puncta_voxels','-v7.3');


headers = cell(size(insitu_transcripts,1),1);
for idx = 1:length(headers)
    p = round(puncta_centroids(idx,:));
    headers{idx} = sprintf('puncta=%i x=%i y=%i z=%i',idx,p(1),p(2),p(3));
end

%fastqstructs = saveExSeqToFastQLike(insitu_transcripts,round(insitu_transcripts_confidence),headers);
%outputfilename = fullfile(params.basecallingResultsDir,sprintf('%s_confidences_phredlike.fastq',params.FILE_BASENAME));
%if exist(outputfilename,'file'); delete(outputfilename); end
%fastqwrite(outputfilename,fastqstructs);

fastqstructs = saveExSeqToFastQLike(insitu_transcripts,round(insitu_transcripts_confidence),headers);
outputfilename = fullfile(params.basecallingResultsDir,sprintf('%s_confidences_fast.fastq',params.FILE_BASENAME));
if exist(outputfilename,'file'); delete(outputfilename); end
fastqwrite(outputfilename,fastqstructs);

%Now create shuffled versions of these basecall rsults
insitu_transcripts_shuffled = zeros(size(insitu_transcripts));
for p_idx = 1:size(insitu_transcripts,1)
    insitu_transcripts_shuffled(p_idx,:) = insitu_transcripts(p_idx,randperm(NUMCALLEDROUNDS));
end

fastqstructs = saveExSeqToFastQLike(insitu_transcripts_shuffled,round(insitu_transcripts_confidence),headers);
outputfilename = fullfile(params.basecallingResultsDir,sprintf('%s_confidences_fast_shuffled.fastq',params.FILE_BASENAME));
if exist(outputfilename,'file'); delete(outputfilename); end
fastqwrite(outputfilename,fastqstructs);


%fastqstructs = saveExSeqToFastQLike(insitu_transcripts_shuffled,round(insitu_transcripts_confidence),headers);
%outputfilename = fullfile(params.basecallingResultsDir,sprintf('%s_confidences_phredlike_shuffled.fastq',params.FILE_BASENAME));
%if exist(outputfilename,'file'); delete(outputfilename); end
%fastqwrite(outputfilename,fastqstructs);

return



