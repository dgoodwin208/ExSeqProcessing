
load('/Users/Goody/Neuro/ExSeq/exseq20170524/exseqautoframe7_transcriptsAndPunctaSet.mat')
loadParameters;
%%


num_paths = 1000;
path_indices = randi(length(transcript_objects),num_paths,1)';


%Pre-initialize the cell arrray

total_number_of_pixels =num_paths*100*20;


fprintf('Makign CSV of %i rows\n',total_number_of_pixels);

output_cell = cell(total_number_of_pixels,1);
ctr = 1;

max_intensity = 10;

for p_idx= 1:length(path_indices)
    
    path_idx = path_indices(p_idx);
    
    
    for rnd_idx = 1:params.NUM_ROUNDS
        
        
        %         puncta_chans = squeeze(puncta_set_normed(:,:,:,rnd_idx,:,p_idx));
        puncta_chans_nonnormed = squeeze(puncta_set(:,:,:,rnd_idx,:,p_idx));
        
        %Take a z-max project Turn the data into X Y C
        %         puncta_chans_zproj = zeros(10,10,3);
        puncta_chans_nonnormed_zproj = zeros(10,10,3);
        for c_idx = 1:4
            %             puncta_chans_zproj(:,:,c_idx) = max(puncta_chans(:,:,:,c_idx),[],3);
            puncta_chans_nonnormed_zproj(:,:,c_idx) = max(puncta_chans_nonnormed(:,:,:,c_idx),[],3);
        end
        
        centroid = transcript_objects{p_idx}.pos;
        [X, Y] = meshgrid(1:10,1:10);
        X = X(:); Y = Y(:);
        
        posX = round(centroid(1)+X(:));
        posY = round(centroid(2)+Y(:));
        posZ = round(ones(length(posY),1)*rnd_idx);
        
        %normalize between 0 and 1
        puncta_chans_nonnormed_zproj = round(puncta_chans_nonnormed_zproj./max(puncta_chans_nonnormed_zproj(:)));
        
        %If we want to view non-normalized pdata:
        rgb_img = makeRGBImageFrom4ChanData(puncta_chans_nonnormed_zproj);
        
        for i = 1:length(posX)
            
            r = rgb_img(X(i),Y(i),1);
            g = rgb_img(X(i),Y(i),2);
            b = rgb_img(X(i),Y(i),3);
            if sum(rgb_img(X(i),Y(i),:))<10
                a=0;
            else
                a = 155;
            end
            
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

%% Make a 3D set of all the puncta

% num_puncta = 1000;
% puncta_indices = randi(length(transcript_objects),num_puncta,1)';
puncta_indices = 1:length(transcript_objects);

%Pre-initialize the cell arrray
%Assume the maximum number of pixels per puncta
total_number_of_pixels =num_puncta*1000;


fprintf('Makign CSV of %i rows\n',total_number_of_pixels);

output_cell = cell(total_number_of_pixels,1);
ctr = 1;

max_intensity = 10;
search_genes = {'NR_046233','NR_002847'};

for p_idx= 1:length(puncta_indices)
    
    path_idx = puncta_indices(p_idx);
    
    %Doesn't matter which puncta round we choose, all volumes are the
    %same
    
    puncta_chans_nonnormed = squeeze(puncta_set(:,:,:,4,:,path_idx));
    
    %Take a z-max project Turn the data into X Y C
    puncta_chans_nonnormed_proj = max(puncta_chans_nonnormed,[],4);
    
    centroid = transcript_objects{path_idx}.pos;
    [X, Y, Z] = meshgrid(1:10,1:10,1:10);
    X = X(:); Y = Y(:); Z = Z(:);
    
    posX = round(centroid(1)+X(:));
    posY = round(centroid(2)+Y(:));
    posZ = round(centroid(3)+Z(:));
    
    
    %   If rainbow
    %     r = randi(255);
    %     g = randi(255);
    %     b = randi(255);
    %If by hamming score:
    
    match_transcript = 0;
    if isfield(transcript_objects{path_idx},'name')
        illumina_name = transcript_objects{path_idx}.name;
    else
        illumina_name='';
    end
    
    for code_idx = 1:length(search_genes)
        if contains(illumina_name,search_genes{code_idx},'IgnoreCase',true)
            if code_idx==1
              r=255;g=0;b=0;  
            else 
             g=255;r=0;b=0;   
            end
            match_transcript=1;
            break;
        end
    end
    
    if ~match_transcript
        g=80;r=80;b=80;  
    end
    
%     if transcript_objects{path_idx}.hamming_score<=1
%         g=255;r=0;b=0;
%     elseif transcript_objects{path_idx}.hamming_score<=3
%         g=0;r=0;b=255;
%     else
%         g=80;r=80;b=80;
%     end
    
    a = 155;
    for i = 1:length(posX)
        
        val = puncta_chans_nonnormed_proj(X(i),Y(i),Z(i));
        if val==0
            continue;
        end
        
        output_cell{ctr} = sprintf('%i,%i,%i,%i,%i,%i,%i,%i\n', posX(i),posY(i),posZ(i),...
            r,g,b,a,4);
        ctr = ctr+1;
    end
    
    
    if mod(p_idx,1000)==0
        fprintf('%i/%i processed\n',p_idx,length(puncta_indices))
    end
end

output_cell(ctr:end)=[];
fprintf('For loop complete\n');

output_csv = strjoin(output_cell,'');

output_file = '/Users/Goody/Coding/of_v0.9.0_osx_release/apps/myApps/ExSeqViewer/bin/punctaIn3D_NR_046233_AND_NR_002847.csv';

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

fprintf('Done!\n')


%% Make a quick 2D image of all the centroids

output_img_allpuncta = zeros(2048,2048);
for t_idx= 1:length(transcript_objects)
    centroid = round(transcript_objects{t_idx}.pos);
    puncta_chans_nonnormed = squeeze(puncta_set(:,:,:,rnd_idx,:,p_idx));
    
    
    puncta_chans_nonnormed_zproj = zeros(10,10,3);
    for c_idx = 1:4

        puncta_chans_nonnormed_zproj(:,:,c_idx) = max(puncta_chans_nonnormed(:,:,:,c_idx),[],3);
    end
    
    output_img_allpuncta(centroid(1)-1:centroid(1)+1,centroid(2)-1:centroid(2)+1) + 50;
    
end
figure; imagesc(output_img_allpuncta,[0 100])
save3DTif_uint16(output_img_allpuncta,'puncta_all.tif');



%%

num_puncta = 10000;

puncta_indices = randi(length(puncta_indices),num_puncta,1)';
% puncta_indices = 1:length(transcript_objects);

%Pre-initialize the cell arrray
%Assume the maximum number of pixels per puncta
total_number_of_pixels =length(puncta_indices)*100;

fprintf('Makign CSV of %i rows\n',total_number_of_pixels);

output_cell = cell(total_number_of_pixels,1);
ctr = 1;

for p_idx= 1:length(puncta_indices)
    
    path_idx = puncta_indices(p_idx);
    
    
    %Doesn't matter which puncta round we choose, all volumes are the
    %same
    
    puncta_chans_nonnormed = squeeze(puncta_set(:,:,:,4,:,path_idx));
    
    %Take a z-max project to urn the data into X Y C
    puncta_chans_nonnormed_proj = max(puncta_chans_nonnormed,[],4);
    
    XY_pixel_mask = max(puncta_chans_nonnormed_proj,[],3);
    
    centroid = transcript_objects{path_idx}.pos;
    [X, Y] = meshgrid(1:10,1:10);
    X = X(:); Y = Y(:); 
    
    posX = round(centroid(1)+X(:));
    posY = round(centroid(2)+Y(:));
    
    a = 155;
    for rnd_idx = 1:params.NUM_ROUNDS

         if transcript_objects{path_idx}.img_transcript(rnd_idx)==1
            g=0;r=0;b=255;
         elseif transcript_objects{path_idx}.img_transcript(rnd_idx)==2
            g=255;r=0;b=0;
         elseif transcript_objects{path_idx}.img_transcript(rnd_idx)==3
            g=0;r=255;b=255;
         else
            g=0;r=255;b=0;
         end
         
        Z=rnd_idx;
        for i = 1:length(posX)

            val = XY_pixel_mask(X(i),Y(i));
            if val==0
                continue;
            end

            output_cell{ctr} = sprintf('%i,%i,%i,%i,%i,%i,%i,%i\n', posX(i),posY(i),Z,...
                r,g,b,a,rnd_idx);
            ctr = ctr+1;
        end
    end
    
    if mod(p_idx,1000)==0
        fprintf('%i/%i processed\n',p_idx,length(puncta_indices))
    end
end

output_cell(ctr:end)=[];
fprintf('For loop complete\n');

output_csv = strjoin(output_cell,'');

output_file = '/Users/Goody/Coding/of_v0.9.0_osx_release/apps/myApps/ExSeqViewer/bin/transcript_stack_random.csv';

fileID = fopen(output_file,'w');
fprintf(fileID,output_csv);
fclose(fileID);

fprintf('Done!\n')

%% Search for the two genes
search_genes = {'NR_046233','NR_002847'};

output_img = zeros(2048,2048,length(search_genes));

for t_idx= 1:length(transcript_objects)
    if isfield(transcript_objects{t_idx},'name')
        illumina_name = transcript_objects{t_idx}.name;
        for code_idx = 1:length(search_genes)
            if contains(illumina_name,search_genes{code_idx},'IgnoreCase',true)
                indices_of_match = [indices_of_match t_idx];
                centroid = round(transcript_objects{t_idx}.pos);
                centroid(1) = min(max(centroid(1),3),2046);
                centroid(2) = min(max(centroid(2),3),2046);
                output_img(centroid(1)-2:centroid(1)+2,centroid(2)-2:centroid(2)+2,code_idx) =...
                    output_img(centroid(1)-2:centroid(1)+2,centroid(2)-2:centroid(2)+2) + 50;
            end
        end
    end
end


for code_idx = 1:length(search_genes)
    save3DTif_uint16(squeeze(output_img(:,:,code_idx)),sprintf('puncta_gene_%s.tif',search_genes{code_idx}));
end


