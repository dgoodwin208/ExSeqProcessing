%%
loadParameters;

% %change the directories if necessary
% params.registeredImagesDir = '/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170904/4_registration-cropped/';
% params.transcriptsResultsDir ='/mp/nas0/ExSeq/AutoSeqHippocampus_results/20170904/5_puncta-extraction/';
% params.FILE_BASENAME ='exseqautoframe7crop';
TARGET_HAMMING_DISTANCE = 0;
RND_IMG = 8; % Which round are we visualizing for the first plots
%
% %Load all the puncta that did not align
% fprintf('Loading puncta and transcripts...');
% load(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_normedroisv12.mat',params.FILE_BASENAME)));
% load(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches.mat',params.FILE_BASENAME)));

% fprintf('DONE\n');
load(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches_objects.mat',params.FILE_BASENAME)))
load(fullfile(params.transcriptResultsDir,sprintf('%s_puncta_rois.mat',params.FILE_BASENAME)))
%%
filename_chan1 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch00'));
filename_chan2 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch01SHIFT'));
filename_chan3 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch02SHIFT'));
filename_chan4 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch03SHIFT'));

fprintf('Images for Round %i...',RND_IMG);
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

%% Generate the positions
scoresAndPositions = cell2mat(cellfun(@(x) [x.hamming_score x.pos], transcript_objects,'UniformOutput',0));
indices = find(scoresAndPositions(:,1)==TARGET_HAMMING_DISTANCE);
indices = indices(randperm(length(indices),100));

sampled_positions = scoresAndPositions(indices,2:4);

%%
%
% hasInitGif = 0;
% giffilename1=sprintf('%s_gridplots_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
% giffilename2=sprintf('%s_puncta_locations_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
%
% if exist(giffilename1,'file')
%     fprintf('DEBUG: %s exists, deleting.\n',giffilename1);
%     delete giffilename1;
% end
%
% if exist(giffilename2,'file')
%     fprintf('DEBUG: %s exists, deleting.\n',giffilename2);
%     delete giffilename2;
% end
%
%
% figure(1)
% set(gcf,'Visible','Off');
% set(gcf,'pos',[  421 5   560   800]);
%
% figure(2)
% set(gcf,'Visible','Off');
%
% scoresAndPositions = cell2mat(cellfun(@(x) [x.hamming_score x.pos], transcript_objects,'UniformOutput',0));
% indices = find(scoresAndPositions(:,1)==TARGET_HAMMING_DISTANCE);
% indices = indices(randperm(length(indices),100));
%
% sampled_positions = scoresAndPositions(indices,2:4);
%
%
% for t_idx = indices'
%
%     if transcript_objects{t_idx}.hamming_score ~=TARGET_HAMMING_DISTANCE
%
%         fprintf('MISTAKE Skipping distance %i\n', transcript_objects{t_idx}.hamming_score);
%         continue;
%     end
%
%
%     puncta = puncta_set_normed(:,:,:,:,:,t_idx);
%     transcript = transcript_objects{t_idx}.img_transcript;
%     visualizeGridPlot(puncta,transcript,params,1)
%
%
%
%     WINDOW=100;
%     clear imgdata4chan
%
%
%     pos = round(transcript_objects{t_idx}.pos);
%     if ~any(pos)
%         fprintf('Skipping pos=[0,0,0] for t_idx=%i\n',t_idx)
%         continue;
%     end
%
%     y_min = max(pos(2)-WINDOW+1,1); y_max = min(pos(2)+WINDOW,size(imgs,1));
%     x_min = max(pos(1)-WINDOW+1,1); x_max = min(pos(1)+WINDOW,size(imgs,2));
%     for c_idx = params.COLOR_VEC
%         imgdata4chan(:,:,c_idx) = imgs(y_min:y_max,x_min:x_max,pos(3),c_idx);
%     end
%
%     rgb_img = makeRGBImageFrom4ChanData(imgdata4chan);
%
%
%     figure(2);
%
%     imshow(rgb_img,'InitialMagnification',250);
%     title(sprintf('Puncta idx %i @ (%i, %i,%i) in Rnd %i',t_idx,pos(2),pos(1),pos(3),RND_IMG));
%     hold on;
%     plot(pos(2)-x_min,pos(1)-y_min,'o','MarkerSize',10,'Color','white');
%     hold off;
%
%
%     drawnow
%
%
%     frame1 = getframe(1);
%     frame2 = getframe(2);
%
%     im1 = frame2im(frame1);
%     im2 = frame2im(frame2);
%
%
%     [imind1,cm1] = rgb2ind(im1,256);
%     [imind2,cm2] = rgb2ind(im2,256);
%
%     if hasInitGif==0
%         imwrite(imind1,cm1,giffilename1,'gif', 'Loopcount',inf);
%         imwrite(imind2,cm2,giffilename2,'gif', 'Loopcount',inf);
%         hasInitGif = 1;
%     else
%         imwrite(imind1,cm1,giffilename1,'gif','WriteMode','append');
%         imwrite(imind2,cm2,giffilename2,'gif','WriteMode','append');
%     end
%
%
%     fprintf('Processed index %i, %i,%i,%i\n',t_idx,pos(1),pos(2),pos(3));
% end

%% Reset and loop over the same indices to create a 4x5 plot of all rounds

WINDOW=100;
hasInitGif = 0;
giffilename1=sprintf('%s_gridplots_norm_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
if exist(giffilename1,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename1);
    delete(giffilename1);
end

giffilename2=sprintf('%s_gridplots_nonnorm_hdist=%i.gif',params.FILE_BASENAME,TARGET_HAMMING_DISTANCE);
if exist(giffilename2,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename2);
    delete(giffilename2);
end

%Create a holder object for the rgb images for all rounds and the 100
%puncta we want to view
all_rgb_regions = cell(params.NUM_ROUNDS,100);
all_chan_regions = cell(params.NUM_ROUNDS,100);
center_point_coords = cell(1,100);
puncta_viewed = [];

madeGridPlotsAndSingleFrame = 0;

figure(1)
set(gcf,'Visible','Off');
set(gcf,'pos',[  421 5   560   800]);
figure(2)
set(gcf,'Visible','Off');
set(gcf,'pos',[  421 5   560   800]);
for rnd_idx = 1:params.NUM_ROUNDS
    
    filename_chan1 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch00'));
    filename_chan2 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch01SHIFT'));
    filename_chan3 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch02SHIFT'));
    filename_chan4 = fullfile(params.registeredImagesDir ,sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,rnd_idx,'ch03SHIFT'));
    
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
        
        y_min = max(pos(2)-WINDOW+1,1); y_max = min(pos(2)+WINDOW,size(imgs,1));
        x_min = max(pos(1)-WINDOW+1,1); x_max = min(pos(1)+WINDOW,size(imgs,2));
        z_min = max(pos(3)-2,1); z_max = min(pos(3)+2,size(imgs,3));
        for c_idx = params.COLOR_VEC
            imgdata4chan(:,:,c_idx) = max(squeeze(imgs(y_min:y_max,x_min:x_max,z_min:z_max,c_idx)),[],3);
        end
        rgb_img = makeRGBImageFrom4ChanData(imgdata4chan);
        all_chan_regions{rnd_idx,ctr} = imgdata4chan;
        all_rgb_regions{rnd_idx,ctr}=rgb_img;
        center_point_coords{1,ctr} = [pos(1)-x_min,pos(2)-y_min];
        %--end making nearby image
        
        %For the first run through, make the gridplots as well
        if ~madeGridPlotsAndSingleFrame
            
            puncta = puncta_set_normed(:,:,:,:,:,t_idx);
            transcript = transcript_objects{t_idx}.img_transcript;
            visualizeGridPlot(puncta,transcript,params,1)
            
            puncta = puncta_set(:,:,:,:,:,t_idx);
            transcript = transcript_objects{t_idx}.img_transcript;
            visualizeGridPlot(puncta,transcript,params,2)
            
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

giffilename3=sprintf('%s_puncta_for_all_images.gif',params.FILE_BASENAME);
if exist(giffilename3,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename3);
    delete(giffilename3);
end

figure(3)
set(gcf,'Visible','Off');
set(gcf,'pos',[ 23, 5, 2600, 1600]);
subplot(4,5,1);
hasInitGif=0;
for puncta_idx = 1:10% size(all_rgb_regions,2)
    
    for rnd_idx = 1:size(all_rgb_regions,1)
        
        subplot(4,5,rnd_idx)
        
        imshow(all_rgb_regions{rnd_idx,puncta_idx},'InitialMagnification',100);
        title(sprintf('rnd%i',rnd_idx));
        hold on;
        pos = center_point_coords{1,puncta_idx};
        
        plot(pos(1),pos(2),'o','MarkerSize',10,'Color','white');
        hold off;
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
end

%% Working on a new version that just shows the four channels independently

clims_perchan = [83, 8397;
    85, 6226;
    23, 3066;
    43, 1500]; %these numbers taken from playing with FIJI

giffilename3=sprintf('%s_gridplot_regions_crop.gif',params.FILE_BASENAME);
if exist(giffilename3,'file')
    fprintf('DEBUG: %s exists, deleting.\n',giffilename3);
    delete(giffilename3);
end

figure(3)
set(gcf,'Visible','Off');
set(gcf,'pos',[ 23, 5,  266*10, 932*10]);

hasInitGif=0;
color_chans = {'b','g','m','r'};
for puncta_idx = 1:10% size(all_chan_regions,2)
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
                title(sprintf('%i',c_idx),'Color','m','FontSize',30)
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
        
        rgb_img = makeRGBImageFrom4ChanData(imgdata4chan);
        axes(ha(subplot_idx));
        imshow(rgb_img); %'InitialMagnification','fit'
        hold on;
        rectangle('Position',[pos(1)-5,pos(2)-5,10,10],...
            'EdgeColor', 'w',...
            'LineWidth', 3,...
            'LineStyle','-'); %                'Curvature',[0.8,0.4],...
        
        hold off;
        
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


