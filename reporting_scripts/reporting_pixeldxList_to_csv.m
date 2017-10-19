
loadParameters;
%Load the pixel_idxs list

filename_centroidsMOD = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels_demerged.mat',params.FILE_BASENAME));
load(filename_centroidsMOD)

filename_output = fullfile(params.punctaSubvolumeDir,sprintf('%s_finalmatches.mat',params.FILE_BASENAME));
load(filename_output,'final_punctapaths');


imgSize = [500 500 101];

%% Make one sample view

RND_IDX = 6;

%Pre-initialize the cell arrray
total_number_of_pixels = 0;
for puncta_idx = 1:length(puncta_voxels{RND_IDX})
    pixels = puncta_voxels{RND_IDX}{puncta_idx};
    total_number_of_pixels =total_number_of_pixels +length(pixels) ;
end
total_number_of_pixels

output_cell = cell(total_number_of_pixels,1);
ctr = 1;
for puncta_idx = 1:length(puncta_voxels{RND_IDX})
    pixels = puncta_voxels{RND_IDX}{puncta_idx};
    [posX, posY, posZ] = ind2sub(imgSize,pixels);
    
    r = randi(255);
    g = randi(255);
    b = randi(255);
    for i = 1:size(pixels,1)
        output_cell{ctr} = sprintf('%i,%i,%i,%i,%i,%i\n', posX(i),posY(i),posZ(i),...
            r,g,b);
        ctr = ctr+1;
    end
    
    if mod(puncta_idx,100)==0
        puncta_idx
    end
end

output_csv = strjoin(output_cell,'');

output_file = '/Users/Goody/Coding/of_v0.9.0_osx_release/apps/myApps/ExSeq3DViewer/bin/randcolor.csv';

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

%% Make a sample view of all rounds in a limited space
% each round in a different color


centerpoint = imgSize/2;
radius_to_show = 25;

%Pre-initialize the cell arrray
total_number_of_pixels = 0;
for rnd_idx = 1:params.NUM_ROUNDS
    for puncta_idx = 1:length(puncta_voxels{rnd_idx})
        centroid = puncta_centroids{rnd_idx}(puncta_idx,:);
        dist = pdist([centerpoint; centroid],'euclidean');
        if dist <= radius_to_show
            pixels = puncta_voxels{rnd_idx}{puncta_idx};
            total_number_of_pixels =total_number_of_pixels +length(pixels) ;
        end
    end
end
fprintf('Makign CSV of %i rows\n',total_number_of_pixels);

output_cell = cell(total_number_of_pixels,1);
ctr = 1;

for rnd_idx = 1:params.NUM_ROUNDS
    
    
    r = randi(255);
    g = randi(255);
    b = randi(255);
    
    for puncta_idx = 1:length(puncta_voxels{rnd_idx})
        centroid = puncta_centroids{rnd_idx}(puncta_idx,:);
        dist = pdist([centerpoint; centroid],'euclidean');
        
        if dist <= radius_to_show
            pixels = puncta_voxels{rnd_idx}{puncta_idx};
            [posX, posY, posZ] = ind2sub(imgSize,pixels);
            for i = 1:size(pixels,1)
                output_cell{ctr} = sprintf('%i,%i,%i,%i,%i,%i,%i\n', posX(i),posY(i),posZ(i),...
                    r,g,b,rnd_idx);
                ctr = ctr+1;
            end
        end
    end
    
    if mod(puncta_idx,100)==0
        puncta_idx
    end
end

output_csv = strjoin(output_cell,'');

output_file = '/Users/Goody/Coding/of_v0.9.0_osx_release/apps/myApps/ExSeq3DViewer/bin/allrounds_spatiallimit.csv';

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

%% Load a few random paths

%Generate random indices
% num_paths = 100;
% path_indices = randi(size(final_punctapaths,1),num_paths,1)';


path_indices = 1:600;% size(final_punctapaths,1);

%Pre-initialize the cell arrray
total_number_of_pixels = 0;
for path_idx = path_indices
    for rnd_idx = 1:params.NUM_ROUNDS
        pixels = puncta_voxels{rnd_idx}{final_punctapaths(path_idx,rnd_idx)};
        total_number_of_pixels =total_number_of_pixels +length(pixels) ;
    end
end
fprintf('Makign CSV of %i rows\n',total_number_of_pixels);


output_cell = cell(total_number_of_pixels,1);
ctr = 1;
for path_idx = path_indices
    r = randi(255);
    g = randi(255);
    b = randi(255);
    
    
    for rnd_idx = 1:params.NUM_ROUNDS
        
        pixels = puncta_voxels{rnd_idx}{final_punctapaths(path_idx,rnd_idx)};
        [posX, posY, posZ] = ind2sub(imgSize,pixels);
        for i = 1:size(pixels,1)
            output_cell{ctr} = sprintf('%i,%i,%i,%i,%i,%i,%i\n', posX(i),posY(i),posZ(i),...
                    r,g,b,rnd_idx);
            ctr = ctr+1;
        end
    end
end
%

output_csv = strjoin(output_cell,'');

output_file = '/Users/Goody/Coding/of_v0.9.0_osx_release/apps/myApps/ExSeqViewer/bin/randompaths.csv';

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

fprintf('Done!')


%% Make a normalized version of puncta_set

puncta_set_normed = zeros(size(puncta_set));
for c = params.COLOR_VEC
    chan_col(:,c) = reshape(puncta_set(:,:,:,:,c,:),[],1);
end
cols_normed = quantilenorm(chan_col);

for c = params.COLOR_VEC
    puncta_set_normed(:,:,:,:,c,:) = reshape(cols_normed(:,c),size(squeeze(puncta_set(:,:,:,:,c,:))));
end


%% Load a few, z-projections and color mapped

%Generate random indices


path_indices = 1:100;

%Pre-initialize the cell arrray and determine the basecalls
total_number_of_pixels = 0;
base_calls_quickzscore=zeros(length(path_indices),params.NUM_ROUNDS);
new_pixelidxlist = {};
chans = zeros(params.PUNCTA_SIZE^3,4);

for p_idx= 1:length(path_indices)
    
%     path_idx = path_indices(p_idx);
    
    for rnd_idx = 1:params.NUM_ROUNDS
        
        %Load and vectorize the puncta_subset
        for c = 1:params.NUM_CHANNELS
            chantemp = puncta_set_normed(:,:,:,rnd_idx,c,p_idx);
            chans(:,c) = chantemp(:)';
        end
        
        sorted_chans = sort(chans,1,'descend');
        %Take the mean of the top 10 values
        scores = mean(sorted_chans(1:10,:),1);
        %and the new baseguess 
        [~, newbaseguess] = max(scores);
        base_calls_quickzscore(p_idx,rnd_idx) = newbaseguess;
       
        total_number_of_pixels = total_number_of_pixels + 100;
    end
end

fprintf('Makign CSV of %i rows\n',total_number_of_pixels);

output_cell = cell(total_number_of_pixels,1);
ctr = 1;

max_intensity = 10;% max(reshape(puncta_set_normed(:,:,:,:,:,path_indices),1,[]));

for p_idx= 1:length(path_indices)
    
    path_idx = path_indices(p_idx);
    
    
    for rnd_idx = 1:params.NUM_ROUNDS
        
        
        puncta_chans = squeeze(puncta_set_normed(:,:,:,rnd_idx,:,p_idx));
        puncta_chans_nonnormed = squeeze(puncta_set(:,:,:,rnd_idx,:,p_idx));
        %Take a z-max project Turn the data into X Y C 
        puncta_chans_zproj = zeros(10,10,3);
        puncta_chans_nonnormed_zproj = zeros(10,10,3);
        for c_idx = 1:4
            puncta_chans_zproj(:,:,c_idx) = max(puncta_chans(:,:,:,c_idx),[],3);
            puncta_chans_nonnormed_zproj(:,:,c_idx) = max(puncta_chans_nonnormed(:,:,:,c_idx),[],3);
        end
        %shift to zero
%         puncta_chans_zproj = puncta_chans_zproj + min(puncta_chans_zproj(:));
%         maxes = zeros(4,1);
%         for c_idx = 1:4
%             t = puncta_chans_zproj(:,:,c_idx);
%             t = t - min(t(:));
%             puncta_chans_zproj(:,:,c_idx) = puncta_chans_zproj(:,:,c_idx)/max(t(:));
%         end
        %get the xyz position
        centroid = puncta_centroids{rnd_idx}(final_punctapaths(path_idx,rnd_idx),:);
        
        [X, Y] = meshgrid(1:10,1:10);
        X = X(:); Y = Y(:);
        
        posX = round(centroid(1)+X(:));
        posY = round(centroid(2)+Y(:));
        posZ = round(ones(length(posY),1)*rnd_idx);
        
        %If we want to view non-normalized pdata:
        rgb_img = makeRGBImageFrom4ChanData(puncta_chans_zproj);
            
        for i = 1:length(posX)
            
            
            
%             pixel = squeeze(puncta_chans_zproj(X(i),Y(i),:));
            
            %If we want to view the normalized data
%             b = min(max(round(255*(pixel(1)/sum(pixel))),0),255);
%             r = min(max(round(255*(pixel(4)/sum(pixel))),0),255);
%             g = min(max(round(255*(pixel(2)/sum(pixel))),0),255);
%             a = min(max(round(255*(sum(pixel)/max_intensity)),0),255);
%             
            
            r = rgb_img(X(i),Y(i),1);
            g = rgb_img(X(i),Y(i),2);
            b = rgb_img(X(i),Y(i),3);
            a = max(mean(rgb_img(X(i),Y(i),:)),max(mean(rgb_img(X(i),Y(i),:)))); 
            %             if sum(rgb_img(X(i),Y(i),:))<10
%                 a=0;
%             else
%                 a = 255;
%             end
            
            output_cell{ctr} = sprintf('%i,%i,%i,%i,%i,%i,%i,%i\n', posX(i),posY(i),posZ(i),...
                    r,g,b,a,rnd_idx);
            ctr = ctr+1;
        end
    end
    
    if mod(p_idx,10)==0
       fprintf('%i/%i processed\n',p_idx,length(path_indices))
    end
end
fprintf('For loop complete\n');
%

output_csv = strjoin(output_cell,'');

output_file = '/Users/Goody/Coding/of_v0.9.0_osx_release/apps/myApps/ExSeqViewer/bin/zstackofrounds.csv';

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

fprintf('Done!\n')
