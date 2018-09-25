%% To load multiple basecalls:

ptr = 1;

for fov_index = 1:10 %1 2 3 4 7 8 9 10] %4 6 7 8
    
    loadParameters;
    params.deconvolutionImagesDir= sprintf('/mp/nas0/ExSeq/AutoSeq2/xy%.2i/1_deconvolution',fov_index);
    params.colorCorrectionImagesDir= sprintf('/mp/nas0/ExSeq/AutoSeq2/xy%.2i/2_color-correction',fov_index);
    params.registeredImagesDir = sprintf('/mp/nas0/ExSeq/AutoSeq2/xy%.2i/4_registration',fov_index);
    params.punctaSubvolumeDir = sprintf('/mp/nas0/ExSeq/AutoSeq2/xy%.2i/5_puncta-extraction',fov_index);
    params.transcriptResultsDir = sprintf('/mp/nas0/ExSeq/AutoSeq2/xy%.2i/6_transcripts',fov_index);
    params.FILE_BASENAME = sprintf('exseqauto-xy%.2i',fov_index);
    params.NUM_ROUNDS = 20;
    
 %        basecallingz;
      %   load(fullfile(params.transcriptResultsDir,sprintf('%s_basecalls_z_dff.mat',params.FILE_BASENAME)))
%     puncta_roicollect_bgincl; delete(gcp('nocreate'));
    %try
    %basecalling;
    %catch%
    %fprintf('Error: likely an old version of puncta used. redoing. \n');
    %puncta_roicollect_bgincl; delete(gcp('nocreate'));
    basecalling;
   % end

%    load(fullfile(params.transcriptResultsDir,sprintf('%s_basecalls_meanpuncta.mat',params.FILE_BASENAME)))
    
    num_reads = size(insitu_transcripts,1);
    
    
    insitu_transcripts_total(ptr:ptr+num_reads-1,:) = insitu_transcripts;
    insitu_transcripts_2ndplace_total(ptr:ptr+num_reads-1,:) = insitu_transcripts_2ndplace;
    insitu_transcripts_confidence_total(ptr:ptr+num_reads-1,:) = insitu_transcripts_confidence_fast;
    base_calls_rawpixel_intensity_total(ptr:ptr+num_reads-1,:,:) = base_calls_rawpixel_intensity;
    base_calls_normedpixel_intensity_total(ptr:ptr+num_reads-1,:,:) = base_calls_normedpixel_intensity;
    puncta_centroids_total(ptr:ptr+num_reads-1,:) = puncta_centroids;
    puncta_voxels_total(ptr:ptr+num_reads-1) = puncta_voxels;
    fovs(ptr:ptr+num_reads-1) = fov_index;
    
    ptr = ptr+num_reads;
    fprintf('Completed round %i\n',fov_index);
end

readlength = size(insitu_transcripts_total,2);

insitu_transcripts = insitu_transcripts_total;
insitu_transcripts_2ndplace = insitu_transcripts_2ndplace_total;
insitu_transcripts_confidence = insitu_transcripts_confidence_total;
base_calls_rawpixel_intensity = base_calls_rawpixel_intensity_total;
base_calls_normedpixel_intensity = base_calls_normedpixel_intensity_total;
puncta_centroids = puncta_centroids_total;
puncta_voxels = puncta_voxels_total;

%%

%LIBRARY_FILE = 'groundtruth_slice_yfp+qs30+locomp.mat';
LIBRARY_FILE = 'groundtruth_dictionary_slice_v6_unique_filtered.mat';
%LIBRARY_FILE = 'groundtruth_dictionary_neuronsv6_unique_filtered.mat';
if ~exist('gtlabels','var')
    load(LIBRARY_FILE);
end

ST_confThresh_changeable = 2;
ST_confThresh_fixed = 7;
ST_editScoreMax = 2;


%% Remove low complexity reads first:

num_puncta = size(insitu_transcripts,1);

H = Entropy(insitu_transcripts');
indices_filtered_entropy = find(H'>1);

fprintf('Discarding %i low entropy reads\n',num_puncta-length(indices_filtered_entropy));
%Then put the values back in (ie don't keep the zeros for base calling)
insitu_transcripts_filtered = insitu_transcripts(indices_filtered_entropy,:);
insitu_transcripts_2ndplace_filtered = insitu_transcripts_2ndplace(indices_filtered_entropy,:);
insitu_transcripts_confidence_filtered = insitu_transcripts_confidence(indices_filtered_entropy,:);
base_calls_rawpixel_intensity_filtered = base_calls_rawpixel_intensity(indices_filtered_entropy,:,:);
base_calls_normedpixel_intensity_filtered = base_calls_normedpixel_intensity(indices_filtered_entropy,:,:);
puncta_centroids_filtered = puncta_centroids(indices_filtered_entropy,:);
puncta_voxels_filtered = puncta_voxels(indices_filtered_entropy);
fovs_filtered = fovs(indices_filtered_entropy);

%% Now remove garbage reads
%Garbage is the mean confidence is less than 5
GARBAGE_READ_MEAN= 5;
indices_keep = find(mean(insitu_transcripts_confidence_filtered,2)>GARBAGE_READ_MEAN);

insitu_transcripts_keep = insitu_transcripts_filtered(indices_keep,:);
insitu_transcripts_2ndplace_keep = insitu_transcripts_2ndplace_filtered(indices_keep,:);
insitu_transcripts_confidence_keep = insitu_transcripts_confidence_filtered(indices_keep,:);
base_calls_rawpixel_intensity_keep = base_calls_rawpixel_intensity_filtered(indices_keep,:,:);
base_calls_normedpixel_intensity_keep= base_calls_normedpixel_intensity_filtered(indices_keep,:,:);
puncta_centroids_keep = puncta_centroids_filtered(indices_keep,:);
puncta_voxels_keep = puncta_voxels_filtered(indices_keep);
fovs_keep = fovs_filtered(indices_keep);

fprintf('Removed %i garbage reads removal \n',...
    length(indices_filtered_entropy) - length(indices_keep));

%% Look for any perfect matches
fprintf('Looking for a perfect match in the %i in situ reads\n',...
    size(insitu_transcripts_keep,1));
perfect_matches = {};
perfect_match_indices = [];
perfect_match_ctr = 1;

searchForPerfects = true;
if searchForPerfects
    tic
    for t = 1:size(insitu_transcripts_confidence_keep)
        img_transcript = insitu_transcripts_keep(t,:);
        img_transcript_2ndplace = insitu_transcripts_2ndplace_keep(t,:);
        perfect_match = find(sum(groundtruth_codes == img_transcript,2)==readlength);
        
        if length(perfect_match)==1
            transcript = struct;
            transcript.img_transcript=img_transcript;
            transcript.img_transcript_confidence=insitu_transcripts_confidence_keep(t,:);
            transcript.img_transcript_absValuePixel=squeeze(base_calls_rawpixel_intensity_keep(t,:,:));
            transcript.img_transcript_normedValuePixel=squeeze(base_calls_normedpixel_intensity_keep(t,:,:));
            transcript.pos = puncta_centroids_keep(t,:);
            transcript.voxels = puncta_voxels_keep{t};
            transcript.hamming_score = 0;
            transcript.fov = fovs_keep(t);
            transcript.known_sequence_matched = groundtruth_codes(perfect_match,:);
            transcript.name = gtlabels{perfect_match};
            
            % Check if it also gets a shuffled hit:
            shuffleindices = randperm(readlength);
            img_transcript_SHUFFLED = img_transcript(shuffleindices);
            img_transcript_2ndplace_SHUFFLED = insitu_transcripts_2ndplace_keep(shuffleindices);
            img_confidence = insitu_transcripts_confidence_keep(t,:);
            img_confidence_SHUFFLED = img_confidence(shuffleindices);
            [matchingIdxSHUFFLED, ~] = shaharTieSeconds(img_transcript_SHUFFLED,img_transcript_2ndplace,img_confidence_SHUFFLED',groundtruth_codes,...
                ST_confThresh_changeable, ST_confThresh_fixed ,ST_editScoreMax);
            %Note if the shuffled version of this transcript got a match!
            %Use the index (1) just in case there are multiple hits
            transcript.shufflehit = matchingIdxSHUFFLED(1)>0;
            
            perfect_match_indices(perfect_match_ctr) = t;
            perfect_matches{perfect_match_ctr} = transcript;
            perfect_match_ctr = perfect_match_ctr+1;
            
            
        end
        
    end
    toc
end
fprintf('%i perfect matches found in situ reads\n',...
    perfect_match_ctr-1);

insitu_transcripts_keep(perfect_match_indices,:) = [];
insitu_transcripts_2ndplace_keep(perfect_match_indices,:) = [];
insitu_transcripts_confidence_keep(perfect_match_indices,:) = [];
base_calls_rawpixel_intensity_keep(perfect_match_indices,:,:) = [];
base_calls_normedpixel_intensity_keep(perfect_match_indices,:,:) = [];
puncta_centroids_keep(perfect_match_indices,:) = [];
puncta_voxels_keep(perfect_match_indices) = [];
fovs_keep(perfect_match_indices) = [];

%% Remove any entries with less than

LOWQUALITY_BASECALL = 5;
LOWQUALITY_NUMBERALLOWABLE = 6;

indices_discard = find(...
    sum(insitu_transcripts_confidence_keep<LOWQUALITY_BASECALL,2)>LOWQUALITY_NUMBERALLOWABLE);

insitu_transcripts_keep(indices_discard,:) = [];
insitu_transcripts_2ndplace_keep(indices_discard,:) = [];
insitu_transcripts_confidence_keep(indices_discard,:) = [];
base_calls_rawpixel_intensity_keep(indices_discard,:,:) = [];
base_calls_normedpixel_intensity_keep(indices_discard,:,:) = [];
puncta_centroids_keep(indices_discard,:) = [];
puncta_voxels_keep(indices_discard) = [];
fovs_keep(indices_discard) = [];
fprintf('Removed %i in situ with too many low quality bases\n',length(indices_discard));


%%



 for ST_confThresh_changeable = [2 3.5 4]
     for ST_confThresh_fixed = [6 7 8 10 ]
         for ST_editScoreMax = 1:2

transcript_objects = cell(size(insitu_transcripts_keep,1),1);
output_cell = cell(size(insitu_transcripts_keep,1),1);

tic
SAMPLE_SIZE = 10000; % size(insitu_transcripts_keep,1);
random_indices = randperm(size(insitu_transcripts_keep,1),SAMPLE_SIZE);
shufflehits = zeros(size(insitu_transcripts_keep,1),1);

hits_pos = zeros(size(insitu_transcripts_keep,1),1);
%parfor p_idx= 1:size(insitu_transcripts_keep,1) %r_idx = 1:length(random_indices)  %
for r_idx = 1:length(random_indices)  %
    
    p_idx = random_indices(r_idx);
    transcript = struct;
    %Search for a perfect match in the ground truth codes
    img_transcript = insitu_transcripts_keep(p_idx,:);
    img_transcript_2ndplace = insitu_transcripts_2ndplace_keep(p_idx,:);
    img_confidence = insitu_transcripts_confidence_keep(p_idx,:);
    
    % Shuffle test!
    shuffleindices = randperm(readlength);
    img_transcript_SHUFFLED = img_transcript(shuffleindices);
    img_transcript_2ndplace_SHUFFLED = insitu_transcripts_2ndplace_keep(shuffleindices);
    img_confidence_SHUFFLED = img_confidence(shuffleindices);
    %Match the real thing
    [matchingIdx, best_score] = shaharTieSeconds(img_transcript,img_transcript_2ndplace_SHUFFLED, img_confidence',groundtruth_codes,...
        ST_confThresh_changeable, ST_confThresh_fixed,ST_editScoreMax);
    %Match a shuffled version as a control
    [matchingIdxSHUFFLED, ~] = shaharTieSeconds(img_transcript_SHUFFLED,img_transcript_2ndplace,img_confidence_SHUFFLED',groundtruth_codes,...
        ST_confThresh_changeable, ST_confThresh_fixed ,ST_editScoreMax);
    %Note if the shuffled version of this transcript got a match!
    %Use the index (1) just in case there are multiple hits
    shufflehits(p_idx) = matchingIdxSHUFFLED(1)>0;
    
    intensities = squeeze(base_calls_rawpixel_intensity_keep(p_idx,:,:));
    
    %Assuming the groundtruth options are de-duplicated
    %Is there a perfect match to the (unique ) best-fit
    transcript.img_transcript=insitu_transcripts_keep(p_idx,:);
    transcript.img_transcript_confidence=img_confidence;
    transcript.img_transcript_absValuePixel=intensities;
    transcript.img_transcript_normedValuePixel=squeeze(base_calls_normedpixel_intensity_keep(p_idx,:,:));
    transcript.pos = puncta_centroids_keep(p_idx,:);
    transcript.fov = fovs_keep(p_idx);
    transcript.voxels = puncta_voxels_keep{p_idx};
    transcript.hamming_score = best_score;
    transcript.shufflehit = shufflehits(p_idx);
    
    if matchingIdx>0
        row_string = sprintf('%i,%s,',p_idx,mat2str(insitu_transcripts_keep(p_idx,:)));
        
        %If it's a unique match, use the name
        transcript.known_sequence_matched = groundtruth_codes(matchingIdx,:);
        transcript.name = gtlabels{matchingIdx};
        
        for t= 1:length(matchingIdx)
            row_string = strcat(row_string,sprintf('%s,',gtlabels{matchingIdx(t)}));
        end
        row_string = sprintf('%s\n',row_string);
        %         fprintf('%s',row_string);
        output_cell{p_idx} = row_string;
        
        hits_pos(p_idx) = 1;
    end
    
    
    transcript_objects{p_idx} = transcript;
    
    %if mod(p_idx,10000) ==0
    %    fprintf('%i/%i matched\n',p_idx,size(insitu_transcripts_keep,1));
    %end
end
toc

%How many are non-shuffled hits?
%How many are shuffled hits?
%How many are BOTH shuffled and non-shuffled hits?
%How many are BOTH shuffled non-hits and non-shuffled non-hits?

output_cell = output_cell(~cellfun('isempty',output_cell));
fp_rate = sum(shufflehits)/sum(shufflehits | hits_pos) ;
fprintf('dff: changable=%i, fixed=%i, editMax=%i, sample size %i: %i %i %f %f\n',...
    ST_confThresh_changeable,ST_confThresh_fixed, ST_editScoreMax, SAMPLE_SIZE, ...
    length(output_cell), sum(shufflehits),sum(shufflehits)/length(output_cell),fp_rate);
%cycle through Fovs to get a breakdown:
transcript_objects = transcript_objects(~cellfun('isempty',transcript_objects));
for f = unique(fovs)
    %Get all the transcript objects for an fov and aligned
    matchindices_allaligned = cell2mat(cellfun(@(x) [isfield(x,'name') && (x.fov==f)], transcript_objects,'UniformOutput',0));
    matchindices_all = cell2mat(cellfun(@(x) [(x.fov==f)], transcript_objects,'UniformOutput',0));
    matchindices_shuffled = cell2mat(cellfun(@(x) [(x.fov==f) && x.shufflehit], transcript_objects,'UniformOutput',0));
    count_allaligned = sum(matchindices_allaligned);
    count_all = sum(matchindices_all);
    count_shuffled = sum(matchindices_shuffled);
    fprintf('\tFOV %i:\t%i puncta,\t%i nonshuf matches,\t%i shuf matches,\t%.2f fp\n',...
        f,count_all,count_allaligned,count_shuffled,count_shuffled/count_allaligned);
end

         end
     end
 end


%%


output_csv = strjoin(output_cell,'');
output_file = fullfile(params.transcriptResultsDir,'xy1-10combinedcodes_dffnoz_meanpucta.csv');

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);
false_hits = sum(shufflehits);

transcript_objects_all = [transcript_objects', perfect_matches];
fprintf('Saved transcript_matches_objects! %i hits w %i shuffledhits \n',length(output_cell)+length(perfect_matches),false_hits  );
%%


save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_dffmedian_noz_meanpuncta.mat','xy1-10')),...
    'ST_confThresh_fixed','ST_confThresh_changeable','ST_editScoreMax','transcript_objects_all',...
    'GARBAGE_READ_MEAN','LOWQUALITY_BASECALL','LIBRARY_FILE','LOWQUALITY_BASECALL','LOWQUALITY_NUMBERALLOWABLE','false_hits','-v7.3');


didaligns = cell2mat(cellfun(@(x) [isfield(x,'name')], transcript_objects_all,'UniformOutput',0));
transcript_objects = transcript_objects_all(didaligns);
save(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_dffmediannoz_alignments.mat','xy1-10')),...
    'ST_confThresh_fixed','ST_confThresh_changeable','ST_editScoreMax','transcript_objects',...
    'GARBAGE_READ_MEAN','LOWQUALITY_BASECALL','LIBRARY_FILE','LOWQUALITY_BASECALL','LOWQUALITY_NUMBERALLOWABLE','false_hits','-v7.3');

% just10s = cell2mat(cellfun(@(x) [x.fov==10], transcript_objects,'UniformOutput',0));
