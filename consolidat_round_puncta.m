
loadParameters;


rnd_string = '5_puncta-extraction/exseqautoframe7_rnd%i_puncta_v2_rois.mat';
%Load sample data to get the number of puncta
load(sprintf(rnd_string,5));

num_puncta = size(puncta_set_rnd,2);

%The new holder for the pixels per punta per round per channel
puncta_set_cell = cell(params.NUM_ROUNDS,params.NUM_CHANNELS,num_puncta);


for rnd_idx = 1:20
    rnd_idx
    load(sprintf(rnd_string,rnd_idx));
    for p_idx = 1:num_puncta
        for c_idx = 1:4
            puncta_set_cell{rnd_idx,c_idx,p_idx} = puncta_set_rnd{c_idx,p_idx};
        end
    end
end

output_string = '5_puncta-extraction/exseqautoframe7_allrounds_puncta_v2_rois.mat';
save(output_string,'puncta_set_cell','-v7.3');
