%%
loadParameters;
%Load all the puncta that did not align
fprintf('Loading puncta and transcripts...');
load(fullfile(params.punctaSubvolumeDir,sprintf('%s_puncta_normedroisv12.mat',params.FILE_BASENAME)));
load(fullfile(params.transcriptResultsDir,sprintf('%s_transcriptmatches.mat',params.FILE_BASENAME)));
fprintf('DONE\n');
%Load a round 
RND_IMG = 8;

filename_chan1 = fullfile('3_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch00'));
filename_chan2 = fullfile('3_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch01SHIFT'));
filename_chan3 = fullfile('3_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch02SHIFT'));
filename_chan4 = fullfile('3_registration',sprintf('%s_round%.03i_%s_registered.tif',params.FILE_BASENAME,RND_IMG,'ch03SHIFT'));

fprintf('Images for Round %i...',RND_IMG);
img = load3DTif(filename_chan1);
imgs = zeros([size(img) 4]);
imgs(:,:,:,1) = img;
img = load3DTif(filename_chan2);
imgs(:,:,:,2) =img;
img = load3DTif(filename_chan3);
imgs(:,:,:,3) = img;
img = load3DTif(filename_chan4);
imgs(:,:,:,4) = img;
clear img;
fprintf('DONE\n');

%%

hasInitGif = 0;
giffilename1='puncta_transcripts_exseq.gif';
giffilename2='puncta_locations.gif';

TARGET_HAMMING_DISTANCE = 3;
output_ctr = 1;

figure(1);
indices = randperm(length(transcript_objects));
for t_idx = indices
    
    if transcript_objects{t_idx}.distance_score-3 ~=TARGET_HAMMING_DISTANCE
        fprintf('Skipping distance %i\n', transcript_objects{t_idx}.distance_score-3);
        continue;
    end
    
    
    puncta = puncta_set_normalsize(:,:,:,:,:,t_idx);
    transcript = transcript_objects{t_idx}.img_transcript;
    visualizeGridPlot(puncta,transcript,params,1)
    
 
    
    WINDOW=100;
    clear imgdata4chan
%     imgdata4chan = zeros(2*WINDOW,2*WINDOW,3);
    pos = transcript_objects{t_idx}.pos;
    if ~any(pos)
        fprintf('Skipping pos=[0,0,0] for t_idx=%i\n',t_idx)
        continue;
    end
    y_min = max(pos(1)-WINDOW+1,1); y_max = min(pos(1)+WINDOW,size(imgs,1));
    x_min = max(pos(2)-WINDOW+1,1); x_max = min(pos(2)+WINDOW,size(imgs,2));
    for c_idx = params.COLOR_VEC
        imgdata4chan(:,:,c_idx) = imgs(y_min:y_max,x_min:x_max,pos(3),c_idx);
    end
    
    rgb_img = makeRGBImageFrom4ChanData(imgdata4chan);
    
    figure(2);
    
    imshow(rgb_img,'InitialMagnification',250);
    title(sprintf('Puncta idx %i @ (%i, %i,%i) in Rnd %i',t_idx,pos(2),pos(1),pos(3),RND_IMG));
    hold on;
    plot(pos(2)-x_min,pos(1)-y_min,'o','MarkerSize',10,'Color','white');
    hold off;
    
    output_ctr = output_ctr+1;
    drawnow
    if hasInitGif==0 %Set the figure sizes etc how we want them
        pause
    end
    
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
    %disp('Paused and Waiting for Enter');
    %pause
    if output_ctr>100
	break
    end
    
end
