

%Load the set of all pixels of all puncta across all rounds from the
filename_pixels = sprintf(fullfile(params.punctaSubvolumeDir,sprintf('%s_punctavoxels.mat',params.FILE_BASENAME)));

load(filename_pixels)

fprintf('Loaded.\n');

%This is the size of the cropped images, corresponding to ~15um pre-ExM
filename_punctaVol = fullfile(params.punctaSubvolumeDir,sprintf('%s_allsummedSummedNorm_puncta.%s',params.FILE_BASENAME,params.IMAGE_EXT));
vol = load3DImage_uint16(filename_punctaVol);
IMG_SIZE = size(vol);
clear vol filename_punctaVol

ILLUMINACORRECTIONFACTOR=-1; 
% The _punctavoxels.mat file contains cell arrays that include the voxel
% locations (stored as 1D indices) and the color values (called puncta_set)
% Here we initialize the _filtered arrays, which will be modified later
puncta_indices_cell_filtered = puncta_indices_cell;
puncta_set_cell_filtered = puncta_set_cell;


%Keep track of funnel numbers:
% [original number of puncta,
% number removed missing bases,
% number aligned,
% shuffled_aligned]
funnel_numbers = zeros(4,1);
funnel_names = {'Segmented amplicons','Present in every round',...
    'Aligned to Barcodes','Column shuffled hits'};

groundtruth_dict = params.GROUND_TRUTH_DICT;
fprintf('Using dicitonary %s \n', groundtruth_dict)
load(groundtruth_dict);

readlength = size(groundtruth_codes,2);

params.ISILLUMINA = false;
%% Unwrap all the puncta into gigantic vectors for quantile norming

%Get the number of filtered puncta
N = length(puncta_indices_cell{1});

funnel_numbers(1) = N;


%Do all the basecalling etc. from one script that has simply been moved out
%of this
%To do the standard color normalization
normalization_qnorm1;
%To do round x color normalization, we call this norm2
% normalization_qnorm2; %This proved to be not as good,


%N is redefined to be the number of filtered puncta
funnel_numbers(2) = N;


%% Basecalling - simply comparing the normalized intensities
%
readlength = size(groundtruth_codes,2);

transcript_objects = {};
% insitu_transcripts_aligned = [];
match_ctr=1;

shuffled_hits = 0;
not_aligned = zeros(size(insitu_transcripts_filtered,1),1);
NMISMATCH=0;
for t = 1:size(insitu_transcripts_filtered,1)
    img_transcript = insitu_transcripts_filtered(t,:);

    %Allow at most mismatch
    perfect_match = find(sum(groundtruth_codes == img_transcript,2)>=(readlength-NMISMATCH));

    % There can only be a unique match in this case
    if length(perfect_match)==1
        transcript = struct;
        transcript.img_transcript=img_transcript;

        voxels = puncta_voxels_filtered{t};
        centroid = puncta_centroids_filtered(t,:);

        transcript.pos = centroid;
        transcript.voxels = voxels;

        transcript.name = gtlabels{perfect_match};
        transcript.error = readlength- sum(groundtruth_codes(perfect_match,:) == img_transcript);
        transcript_objects{match_ctr} = transcript;

        % Create an aligned version of the insitu_transcript
%         insitu_transcripts_aligned(match_ctr,:) = img_transcript;
        match_ctr = match_ctr+1;

    else
        if length(perfect_match)>1
            fprintf('Found %i matches with in Hamming-1 so discarding\n',length(perfect_match));
        end
        not_aligned(t) = 1;
    end

    %Shuffle transcripts to get a false pos rate. Column wise shuffling:
    img_transcript_shuffled = diag(insitu_transcripts_filtered(randperm(size(insitu_transcripts_filtered,1),readlength),1:readlength))';

    perfect_match = find(sum(groundtruth_codes == img_transcript_shuffled,2)>=(readlength-NMISMATCH));
    if length(perfect_match)==1
        shuffled_hits = shuffled_hits+1;
    end
end

% Get all the gene names so we can make a (sorted!) histogram
insitu_genes = cell(length(transcript_objects),1);
for t = 1:length(transcript_objects)
    insitu_genes{t} = transcript_objects{t}.name;
end
insitu_genes = categorical(insitu_genes);
figure; histogram(insitu_genes,'DisplayOrder','descend')
title(sprintf('%i alignments',match_ctr-1));

fprintf('Of %i transcripts, %i matches\n',size(insitu_transcripts_filtered,1),length(transcript_objects));

% Note the
funnel_numbers(3) = length(transcript_objects);
funnel_numbers(4) = shuffled_hits;

save(fullfile(params.basecallingResultsDir,sprintf('%s_results.mat',params.FILE_BASENAME)),'transcript_objects','funnel_numbers', 'groundtruth_dict');
