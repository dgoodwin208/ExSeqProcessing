loadParameters;
load(fullfile(params.rajlabDirectory,'transcriptsv9_punctameannormed.mat'));
load(fullfile(params.rajlabDirectory,'transcripts_probfiltered.mat'));

[unique_transcripts, ia, ic] = unique(transcripts_probfiltered,'rows');

barcodes_count = size(unique_transcripts,1);

labels = {};
for i = 1:barcodes_count
    transcript_string = '';
    for c = 1:size(unique_transcripts,2)
        transcript_string(c) = num2str(unique_transcripts(i,c));
    end
    labels{i}=transcript_string;
end

%%
labelstring_per_puncta = cell(length(transcripts_probfiltered),1);
for i = 1:length(labelstring_per_puncta)
    labelstring_per_puncta{i} = labels{ic(i)};
end


save('barcodes_output.mat','pos','labelstring_per_puncta','transcripts_probfiltered');

