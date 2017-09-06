function makePunctaVolumes2(rnd_idx)
%Load the params file which includes all parameters specific to the
%experiment
loadParameters;

%Get the filtered puncta coords from analyzePuncta.m script
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_centroid_collections.mat',params.FILE_BASENAME)));

REF_IDX = 5;

%Many of the centroid locations are going to be zero, so how many are
%there?
filtered_centroid_mask = centroid_collections(:,REF_IDX,1)==0;
num_puncta = size(centroid_collections,1)- sum(filtered_centroid_mask);

reference_centroids = centroid_collections(filtered_centroid_mask,REF_IDX,:);
%Define a puncta_set object that can be parallelized
%puncta_set_cell = cell(params.NUM_ROUNDS,1);

%for rnd_idx = 1:params.NUM_ROUNDS
    
    filename_chan1 = fullfile('4_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch00'));
    filename_chan2 = fullfile('4_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch01SHIFT'));
    filename_chan3 = fullfile('4_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch02SHIFT'));
    filename_chan4 = fullfile('4_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch03SHIFT'));
    
    clear imgs img;  
    fprintf('Images for Round %i...',rnd_idx);
    img = load3DTif_uint16(filename_chan1);
    imgs = zeros([size(img) 4]);
    imgs(:,:,:,1) = img;
    img = load3DTif_uint16(filename_chan2);
    imgs(:,:,:,2) =img;
    img = load3DTif_uint16(filename_chan3);
    imgs(:,:,:,3) = img;
    img = load3DTif_uint16(filename_chan4);
    imgs(:,:,:,4) = img;
    clear img;
    fprintf('DONE\n');
    
    
    %puncta_set_cell{rnd_idx} = cell(params.NUM_CHANNELS,num_puncta);
    puncta_set_rnd = cell(params.NUM_CHANNELS,num_puncta);

    %Loop over all the puncta
    %Because of the current indexing scheme, there will be about 90% zeros so
    %the way to filter this is check the ref idx location to be zeros
    puncta_ctr = 1;
    
    good_puncta_indices = find(squeeze(centroid_collections(:,REF_IDX,1))>0);
    for p_idx = good_puncta_indices' %1:size(centroid_collections,1)
        %if centroid_collections(p_idx,REF_IDX,1)==0
        %    continue;
        %end
        
        c = round(centroid_collections(p_idx,rnd_idx,:));

        %correct the Z value manually for now
        c(3) = round(c(3)*(.17/.5));
        imgdims = size(squeeze(imgs(:,:,:,1)));
        ymin = max(c(1) - params.PUNCTA_SIZE/2 + 1,1);
        ymax = min(c(1) + params.PUNCTA_SIZE/2,imgdims(1));
        
        xmin = max(c(2) - params.PUNCTA_SIZE/2 + 1,1);
        xmax = min(c(2) + params.PUNCTA_SIZE/2,imgdims(2));

        zmin = max(c(3) - params.PUNCTA_SIZE/2 + 1,1);
        zmax = min(c(3) + params.PUNCTA_SIZE/2,imgdims(3));


        y_indices = ymin:ymax;
        x_indices = xmin:xmax;
        z_indices = zmin:zmax;
    
     
       for c_idx = params.COLOR_VEC
            puncta_set_rnd{c_idx,puncta_ctr} = imgs(y_indices,x_indices,z_indices,c_idx);
        end
        
        puncta_ctr = puncta_ctr +1;
        if mod(puncta_ctr,100)==0
		fprintf('Processed %i puncta', puncta_ctr);
        end
    end
%end
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_rnd%i_puncta_v2_rois.mat',params.FILE_BASENAME,rnd_idx)),...
    'puncta_set_rnd','reference_centroids','good_puncta_indices','-v7.3');

return

disp('reducing processed puncta')
%Remove all puncta from the set that are too close to the boundary of the
%image
puncta_set = puncta_set(:,:,:,:,:,num_puncta);

% reduction of parfor
for exp_idx = 1:params.NUM_ROUNDS
    for puncta_idx = 1:num_puncta_filtered
        for c_idx = params.COLOR_VEC
            if ~isequal(size(puncta_set_cell{exp_idx}{c_idx,puncta_idx}), [0 0])
                puncta_set(:,:,:,exp_idx,c_idx,puncta_idx) = puncta_set_cell{exp_idx}{c_idx,puncta_idx};
            end
        end
    end
end

disp('saving files from makePunctaVolumes2')
%just save puncta_set
save(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_v2_rois.mat',params.FILE_BASENAME)),...
    'puncta_set','reference_centroids','-v7.3');


