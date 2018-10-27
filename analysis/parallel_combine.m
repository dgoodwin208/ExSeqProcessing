%Split up a directory of registered images into _1, _2, etc.
loadParameters;
params.registeredImagesDir = '/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170904/4_registration';
params.FILE_BASENAME= 'exseqautoframe7';

%how many subsections to calculate the descriptors?
params.ROWS_DESC = 3;
params.COLS_DESC = 3;
params.OVERLAP = .1;

filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_registered.tif',...
    params.FILE_BASENAME,1,params.CHAN_STRS{1}));
img = load3DTif_uint16(filename);

%chop the image up into grid
tile_upperleft_y = floor(linspace(1,size(img,1),params.ROWS_DESC+1));
tile_upperleft_x = floor(linspace(1,size(img,2),params.COLS_DESC+1));


root_registration_directory = params.registeredImagesDir;
root_basename = params.FILE_BASENAME;
tile_counter = 0;

puncta_set_cell = cell(params.ROWS_DESC*params.COLS_DESC,1);
transcript_objects_cell = cell(params.ROWS_DESC*params.COLS_DESC,1);
total_number_of_puncta = zeros(params.ROWS_DESC*params.COLS_DESC,1);

for x_idx=1:params.COLS_DESC
    for y_idx=1:params.ROWS_DESC
        
        % get region, indexing column-wise
        ymin = tile_upperleft_y(y_idx);
        ymax = tile_upperleft_y(y_idx+1);
        xmin = tile_upperleft_x(x_idx);
        xmax = tile_upperleft_x(x_idx+1);
        
        %Adjust by the overlap
        ymin_overlap = floor(max(tile_upperleft_y(y_idx)-(params.OVERLAP/2)*(ymax-ymin),1));
        xmin_overlap = floor(max(tile_upperleft_x(x_idx)-(params.OVERLAP/2)*(xmax-xmin),1));
        
        
        tile_counter = tile_counter+1;
        
        directory_to_process = fullfile(root_registration_directory,sprintf('subpiece%i',tile_counter));
        
        disp(['Running on row ' num2str(y_idx) ' and col ' num2str(x_idx) ]);
        
        params.registeredImagesDir = directory_to_process;
        params.punctaSubvolumeDir = directory_to_process;
        params.basecallingResultsDir = directory_to_process;
        params.FILE_BASENAME = sprintf('%s_%i',root_basename,tile_counter);
        
        %load transcript_objects
        load(fullfile(directory_to_process,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)));
        %loop over all the objects and shift back into global coordinates
        for idx = 1:length(transcript_objects)
%             transcript_objects{idx}.pos = transcript_objects{idx}.pos + [ymin_overlap-1, xmin_overlap-1, 0];
            %Testing if it's a matlab improps hidden transpose
            transcript_objects{idx}.pos = transcript_objects{idx}.pos + [xmin_overlap-1,ymin_overlap-1, 0];
        end
        
        %load puncta_set
        load(fullfile(directory_to_process,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)));
        
        %Load puncta information - puncta_centroids, puncta_voxels,puncta_baseguess
        filename_centroidsMOD = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels_demerged.mat',params.FILE_BASENAME));
        
        
        
        puncta_set_cell{tile_counter} = puncta_set;
        transcript_objects_cell{tile_counter} = transcript_objects;
        total_number_of_puncta(tile_counter) = size(puncta_set,6);
        fprintf('Processed %i puncta \n',size(puncta_set,6));
    end
end

size_of_all_puncta = size(puncta_set_cell{tile_counter});
size_of_all_puncta(6) = sum(total_number_of_puncta);

puncta_set_complete = zeros(size_of_all_puncta);
transcript_objects_complete = cell(sum(total_number_of_puncta),1);

for tile_counter = 1:params.ROWS_DESC*params.COLS_DESC
    %start_idx is 1 for tile_counter=1 and the
    start_idx = sum(total_number_of_puncta(1:(tile_counter-1)))+1;
    end_idx = sum(total_number_of_puncta(1:tile_counter));
    fprintf('Noting indices %i to  %i\n',start_idx,end_idx);
    puncta_set_complete(:,:,:,:,:,start_idx:end_idx) = puncta_set_cell{tile_counter};
    transcript_objects_complete(start_idx:end_idx) = transcript_objects_cell{tile_counter};
end

puncta_set = uint16(puncta_set_complete);
transcript_objects= transcript_objects_complete;
save(fullfile(root_registration_directory,sprintf('%s_transcriptsAndPunctaSet.mat',root_basename)),...
    'puncta_set','transcript_objects','-v7.3');


%% Convert all the data into zscores (very cheap base calling)
puncta_set_normed = zeros(size(puncta_set));
clear chan_col; %Just in case, otherwise the for loop can error.
for c = params.COLOR_VEC
    chan_col(:,c) = reshape(puncta_set(:,:,:,:,c,:),[],1);
end

% cols_normed = quantilenorm(chan_col);
cols_normed = zscore(single(chan_col));

for c = params.COLOR_VEC
    puncta_set_normed(:,:,:,:,c,:) = reshape(cols_normed(:,c),size(squeeze(puncta_set(:,:,:,:,c,:))));
end

save(fullfile(root_registration_directory,sprintf('%s_normalizedPunctaSet.mat',root_basename)),...
    'puncta_set','transcript_objects','-v7.3');


% %% And make a test image
% output_img = zeros(size(img,1),size(img,2));
% for t_idx= 1:length(transcript_objects)
%     centroid = round(transcript_objects{t_idx}.pos);
% %     centroid = centroid([2 1 3]);
%     centroid(1) = min(max(centroid(1),2),2047);
%     centroid(2) = min(max(centroid(2),2),2047);
%     output_img(centroid(1)-1:centroid(1)+1,centroid(2)-1:centroid(2)+1) =...
%         output_img(centroid(1)-1:centroid(1)+1,centroid(2)-1:centroid(2)+1) + 50;
% end
% figure; imagesc(output_img,[0 100])
% save3DTif_uint16(output_img,fullfile(root_registration_directory,'wtfimg.tif'));
