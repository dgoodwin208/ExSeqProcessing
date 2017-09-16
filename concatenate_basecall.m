
% loadParameters;

% load('5_puncta-extraction/exseqautoframe7_allrounds_puncta_v2_rois.mat');

%For each round and each channel, we will calculate the mean value and std
%dev. Then when we do color-wise comparisons, we can convert the value
%using these numbers
means = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
stds = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);

num_puncta = size(puncta_set,6);
for rnd_idx = 1:params.NUM_ROUNDS
    
    rnd_idx
    %Pre-allocate the values to somethign impossibel and filter out later
    color_vec = zeros(1,1000*num_puncta)-1;
    ctr = 1;
    for c_idx = 1:params.NUM_CHANNELS
        for p_idx = 1:num_puncta
            pixels = puncta_set(:,:,:,rnd_idx,c_idx,p_idx);
            pixels = pixels(:); %linearize pixels
            pixels = pixels(pixels>0); %ignore the zero padding
            if isempty(pixels)
                pixels = [0];
            end
            color_vec(ctr:ctr+length(pixels)-1) = pixels;
            ctr = ctr+length(pixels);
        end
        
        %Clean the pre-allocated vector get the used values
        color_vec_cleaned = color_vec(color_vec>-1);
        means(rnd_idx,c_idx) = mean(color_vec_cleaned);
        stds(rnd_idx,c_idx) = std(color_vec_cleaned);
    end
end

%%
transcripts = zeros(num_puncta,params.NUM_ROUNDS);
transcripts_confidence = zeros(num_puncta,params.NUM_ROUNDS);
% transcripts_calls = zeros(num_puncta^3,params.NUM_ROUNDS,size(puncta_set,6));
for p_idx = 1:num_puncta
    puncta_centers_to_compare = zeros(params.NUM_ROUNDS,params.NUM_CHANNELS);
    for rnd_idx = 1:params.NUM_ROUNDS
        for c_idx = 1:params.NUM_CHANNELS
            
            pixels = puncta_set(:,:,:,rnd_idx,c_idx,p_idx);
            pixels = pixels(:); %linearize pixels
            pixels = pixels(pixels>0); %ignore the zero padding
            if isempty(pixels)
                pixels = [0];
            end
            
            imgdims = size(pixels);
            
            pixels_normed = (pixels-means(rnd_idx,c_idx))/stds(rnd_idx,c_idx);
            
            puncta_centers_to_compare(rnd_idx,c_idx) = mean(pixels_normed(:));
        end
        
        [vals,indices] = sort(squeeze(puncta_centers_to_compare(rnd_idx,:)),'descend');
        
        transcripts(p_idx,rnd_idx) = indices(1);
        transcripts_confidence(p_idx,rnd_idx) = vals(1)/(vals(1)+vals(2));
    end
   
    if mod(p_idx,1000) ==0
        fprintf('%i/%i called\n',p_idx,num_puncta);
    end

end

save('quicktsettranscripts.mat','transcripts','transcripts_confidence');



