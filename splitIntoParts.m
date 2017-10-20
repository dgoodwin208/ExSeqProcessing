
%Randomly permute the 
loadParameters;

SPLIT_FACTOR = 6;
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois_oversize.mat',params.FILE_BASENAME)))

num_puncta = size(puncta_set,6);

rand_indices = randperm(num_puncta);

split_indices = linspace(1,length(rand_indices),SPLIT_FACTOR+1);

for split_piece = 1:SPLIT_FACTOR
   start_idx = split_indices(split_piece);
   end_idx = split_indices(split_piece+1)-1;
   
   indices_for_this_part = rand_indices(start_idx:end_idx);
   save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois_oversize_indices%i.mat',params.FILE_BASENAME,split_piece)),'indices_for_this_part')
end
