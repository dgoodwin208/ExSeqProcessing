%%
loadParameters;

% %change the directories if necessary
% params.registeredImagesDir = '/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170904/4_registration-cropped/';
% params.transcriptsResultsDir ='/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170904/5_puncta-extraction/';
% params.FILE_BASENAME ='exseqautoframe7crop';
TARGET_HAMMING_DISTANCE = 1;
NN_MINIMUM_DISTANCE = 8;
QUERY_NUMBER = 25;
load(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objectsMODIFIED.mat',params.FILE_BASENAME)))
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)))

transcript_objects = transcript_objects_new;
%%

%% Generate the positions
% If we want to sample by spacing and hamming score
scoresAndPositions = cell2mat(cellfun(@(x) [x.hamming_score x.pos x.nn_distance], transcript_objects,'UniformOutput',0));

% If we want to sample by shannon entropy:
transcripts = cell2mat(cellfun(@(x) [x.img_transcript], transcript_objects,'UniformOutput',0));
%h = Entropy(transcripts(:,4:20)');
%indices = find(scoresAndPositions(:,1)==TARGET_HAMMING_DISTANCE & h'>1);
%indices = indices(randperm(length(indices),QUERY_NUMBER));
reprocessedFlags = cell2mat(cellfun(@(x) [x.reprocessed], transcript_objects,'UniformOutput',0));
indices = find(scoresAndPositions(:,1)<=TARGET_HAMMING_DISTANCE & reprocessedFlags);
indices = indices(randperm(length(indices),QUERY_NUMBER));

% indices = find(h>1);
% indices = indices(randperm(length(indices),QUERY_NUMBER));

% sampled_positions = scoresAndPositions(indices,2:4);

puncta_set_normed = zeros(size(puncta_set));
clear chan_col; %Just in case, otherwise the for loop can error.
for c = params.COLOR_VEC
    chan_col(:,c) = reshape(puncta_set(:,:,:,:,c,:),[],1);
end

% cols_normed = quantilenorm(chan_col);
cols_normed = zscore(chan_col);

for c = params.COLOR_VEC
    puncta_set_normed(:,:,:,:,c,:) = reshape(cols_normed(:,c),size(squeeze(puncta_set(:,:,:,:,c,:))));
end


%Name the normalized gridplot file
% giffilename1=sprintf('%s_gridplots_norm_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
giffilename1=sprintf('%s_gridplots_norm_entropy>1.gif',params.FILE_BASENAME);
%Name the non-normalized gridplot file
% giffilename2=sprintf('%s_gridplots_nonnorm_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
giffilename2=sprintf('%s_gridplots_nonnorm_entropy>1.gif',params.FILE_BASENAME);
%Name the extended grid plot file
% giffilename3=sprintf('%s_gridplot_regions_crop_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
giffilename3=sprintf('%s_gridplot_regions_crop_entropy>1.gif',params.FILE_BASENAME);
%% Reset and loop over the same indices to create a 4x5 plot of all rounds

WINDOW=100;
hasInitGif = 0;

if exist(giffilename1,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename1);
    delete(giffilename1);
end


if exist(giffilename2,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename2);
    delete(giffilename2);
end

%Create a holder object for the rgb images for all rounds and the QUERY_NUMBER
%puncta we want to view
all_rgb_regions = cell(params.NUM_ROUNDS,QUERY_NUMBER);
all_chan_regions = cell(params.NUM_ROUNDS,QUERY_NUMBER);
center_point_coords = cell(1,QUERY_NUMBER);
puncta_viewed = [];

madeGridPlotsAndSingleFrame = 0;

figure(1)
set(gcf,'Visible','Off');
set(gcf,'pos',[  421 5   560   800]);
figure(2)
set(gcf,'Visible','Off');
set(gcf,'pos',[  421 5   560   800]);
for rnd_idx = 1:params.NUM_ROUNDS
    
    filename_chan1 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_affine.tif',params.FILE_BASENAME,rnd_idx,'ch00'));
    filename_chan2 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_affine.tif',params.FILE_BASENAME,rnd_idx,'ch01SHIFT'));
    filename_chan3 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_affine.tif',params.FILE_BASENAME,rnd_idx,'ch02SHIFT'));
    filename_chan4 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_affine.tif',params.FILE_BASENAME,rnd_idx,'ch03SHIFT'));
    
    fprintf('Loading images for Round %i...',rnd_idx);
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
    
    
    for ctr = 1:length(indices)
        t_idx = indices(ctr);
        
        %-----Load RGB regions for later making into region images
        clear imgdata4chan
        pos = round(transcript_objects{t_idx}.pos);
        
        y_min = max(pos(1)-WINDOW+1,1); y_max = min(pos(1)+WINDOW,size(imgs,1));
        x_min = max(pos(2)-WINDOW+1,1); x_max = min(pos(2)+WINDOW,size(imgs,2));
        z_min = max(pos(3)-2,1); z_max = min(pos(3)+2,size(imgs,3));
        for c_idx = params.COLOR_VEC
            imgdata4chan(:,:,c_idx) = max(squeeze(imgs(y_min:y_max,x_min:x_max,z_min:z_max,c_idx)),[],3);
        end
        rgb_img = makeRGBImageFrom4ChanData(imgdata4chan);
        all_chan_regions{rnd_idx,ctr} = imgdata4chan;
        all_rgb_regions{rnd_idx,ctr}=rgb_img;
        center_point_coords{1,ctr} = [pos(2)-x_min,pos(1)-y_min];
        %--end making nearby image
        
        %For the first run through, make the gridplots as well
        if ~madeGridPlotsAndSingleFrame
            
            puncta = puncta_set_normed(:,:,:,:,:,t_idx);
            transcript = transcript_objects{t_idx}.img_transcript;
            visualizeGridPlot(puncta,transcript_objects{t_idx},params,1)
            
            puncta = puncta_set(:,:,:,:,:,t_idx);
            transcript = transcript_objects{t_idx}.img_transcript;
            visualizeGridPlot(puncta,transcript_objects{t_idx},params,2)
            
            drawnow
            
            frame1 = getframe(1);
            frame2 = getframe(2);
            im1 = frame2im(frame1);
            im2 = frame2im(frame2);
            [imind1,cm1] = rgb2ind(im1,256);
            [imind2,cm2] = rgb2ind(im2,256);
            
            if hasInitGif==0
                imwrite(imind1,cm1,giffilename1,'gif', 'Loopcount',inf);
                imwrite(imind2,cm2,giffilename2,'gif', 'Loopcount',inf);
                hasInitGif = 1;
            else
                imwrite(imind1,cm1,giffilename1,'gif','WriteMode','append');
                imwrite(imind2,cm2,giffilename2,'gif','WriteMode','append');
            end
            fprintf('%i: Processed index %i, %i,%i,%i\n',ctr, t_idx,pos(1),pos(2),pos(3));
        end
        
    end %finish looping over puncta
    
    if ~madeGridPlotsAndSingleFrame
        fprintf('Completed grid plots\n');
        madeGridPlotsAndSingleFrame = 1;
    end
    
end

save(fullfile(params.transcriptResultsDir,sprintf('%s_chanimagesforpuncta_hdist=%i.mat',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE)),'all_chan_regions','indices');

%% Now actually make the image looping over the cell arrays of the rGB images

% giffilename3=sprintf('%s_puncta_for_all_images.gif',params.FILE_BASENAME);
% if exist(giffilename3,'file')
%     fprintf('DEBUG: %s exists, deleting.\n',giffilename3);
%     delete(giffilename3);
% end

%figure(3)
%set(gcf,'Visible','Off');
%set(gcf,'pos',[ 23, 5, 2600, 1600]);
%subplot(4,5,1);
%hasInitGif=0;
%for puncta_idx = 1:10% size(all_rgb_regions,2)
%    
%    for rnd_idx = 1:size(all_rgb_regions,1)
%        
%        subplot(4,5,rnd_idx)
%        
%        imshow(all_rgb_regions{rnd_idx,puncta_idx},'InitialMagnification',100);
%        title(sprintf('rnd%i',rnd_idx));
%        hold on;
%        pos = center_point_coords{1,puncta_idx};
%        
%        plot(pos(1),pos(2),'o','MarkerSize',10,'Color','white');
%        hold off;
%    end
    
%    frame = getframe(3);
%    im = frame2im(frame);
%    [imind1,cm1] = rgb2ind(im,256);
    
%    if hasInitGif==0
%        imwrite(imind1,cm1,giffilename3,'gif', 'Loopcount',inf);
%        hasInitGif = 1;
%    else
%        imwrite(imind1,cm1,giffilename3,'gif','WriteMode','append');
%    end
%end

%% Working on a new version that just shows the four channels independently

clims_perchan = [83, 8397;
    85, 6226;
    23, 3066;
    43, 1500]; %these numbers taken from playing with FIJI


if exist(giffilename3,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename3);
    delete(giffilename3);
end

figure(3)
set(gcf,'Visible','Off');
set(gcf,'pos',[ 23, 5,  266*10, 932*10]);

hasInitGif=0;
color_chans = {'b','g','m','r'};
for puncta_idx = 1:QUERY_NUMBER %size(all_chan_regions,2)
    transcript_object = transcript_objects{indices(puncta_idx)};
    clf('reset')
    ha = tight_subplot(params.NUM_ROUNDS,params.NUM_CHANNELS+1,zeros(params.NUM_ROUNDS,2)+.01);
    
    
    
    subplot_idx = 1;
    pos = center_point_coords{1,puncta_idx};
    
    for rnd_idx = 1:params.NUM_ROUNDS
        imgdata4chan = all_chan_regions{rnd_idx,puncta_idx};
        
        for c_idx = 1:params.NUM_CHANNELS
            clims = [clims_perchan(c_idx,1),clims_perchan(c_idx,2)];
            
            %Get the subplot index using the tight_subplot system
            
            axes(ha(subplot_idx));
            imagesc(imgdata4chan(:,:,c_idx),clims);
            axis off;
            
            
            hold on;
            rectangle('Position',[pos(1)-5,pos(2)-5,10,10],...
                'EdgeColor', color_chans{c_idx},...
                'LineWidth', 3,...
                'LineStyle','-'); %                'Curvature',[0.8,0.4],...
            
            hold off;
            
            
            if numel(transcript_object.img_transcript)>1 && c_idx==transcript_object.img_transcript(rnd_idx)
                
                title(sprintf('%.0f',transcript_object.img_transcript_absValuePixel(rnd_idx)),'Color','m','FontSize',30)
            end
            
            
            
            if c_idx==1 && rnd_idx ==1
                origpos = round(transcript_object.pos);
                position_string = sprintf('pos=(%i,%i,%i)',origpos(2),origpos(1),origpos(3));
                text(-0.0,10.,position_string,'rotation',90,'FontSize',20)
                axis tight;
            elseif c_idx==1
                position_string = sprintf('%i',rnd_idx);
                text(-0.0,10.,position_string,'rotation',90,'FontSize',20)
                axis tight;
            else
                axis off;
            end
            
            colormap gray
            subplot_idx = subplot_idx+1;
        end
        
        %Add a fifth column to visualize the puncta in RGB
        %Get the subplot index using the tight_subplot system
        
        rgb_img = makeRGBImageFrom4ChanData(imgdata4chan,clims_perchan);
        axes(ha(subplot_idx));
        imshow(uint8(rgb_img)); %'InitialMagnification','fit'
        hold on;
        rectangle('Position',[pos(1)-5,pos(2)-5,10,10],...
            'EdgeColor', 'w',...
            'LineWidth', 3,...
            'LineStyle','-'); %                'Curvature',[0.8,0.4],...
        
        hold off;
        if ~isfield(transcript_object,'img_transcript_confidence')
             title('N/A','Color','m','FontSize',30)
        else
        title(sprintf('%.02f',transcript_object.img_transcript_confidence(rnd_idx)),'Color','m','FontSize',30)
        end

        subplot_idx = subplot_idx+1;
        
    end
    
    
    
    frame = getframe(3);
    im = frame2im(frame);
    [imind1,cm1] = rgb2ind(im,256);
    
    if hasInitGif==0
        imwrite(imind1,cm1,giffilename3,'gif', 'Loopcount',inf);
        hasInitGif = 1;
    else
        imwrite(imind1,cm1,giffilename3,'gif','WriteMode','append');
    end
    fprintf('Processed puncta %i\n',puncta_idx);
end



