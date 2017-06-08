%Written for the first splintr barcode dataset from Oz
% FOLDER_NAME = 'ExSeqAutoSlice';
FILEROOT_NAME = 'PrimerNPCCCframe7';

DIRECTORY = '/mp/nas0/ExSeq';

offsets3D = [8,8,5]; %X,Y,Z offsets for calcuating the difference
roundnum = 1;
BEAD_ZSTART = 120;

    %Load all channels, normalize them, calculate the cross corr of 
    %channels 1-3 vs 4
    chan1 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch00.tif',FILEROOT_NAME,roundnum)));
    chan2 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch01.tif',FILEROOT_NAME,roundnum)));
    chan3 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch02.tif',FILEROOT_NAME,roundnum)));
    chan4 = load3DTif(fullfile(DIRECTORY,sprintf('%s_round%i_ch03.tif',FILEROOT_NAME,roundnum)));
    
    chan1_beads = chan1(:,:,BEAD_ZSTART:end);
    chan4_beads = chan4(:,:,BEAD_ZSTART:end);
    %%
    xcorr_scores = zeros(offsets3D); 
    for z = -1*offsets3D(3):offsets3D(3)
        for y = -1*offsets3D(2):offsets3D(2)
            for x = -1*offsets3D(1):offsets3D(1)
                chan4_beads_shift = circshift(chan4_beads,x,1);
                chan4_beads_shift = circshift(chan4_beads_shift,y,2);
                chan4_beads_shift = circshift(chan4_beads_shift,z,3);
                xcorr_scores(x+offsets3D(1)+1,y+offsets3D(2)+1,z+offsets3D(3)+1) = ...
                     sum(chan1_beads(:).*chan4_beads_shift(:));
                disp(sprintf('Calculated z=%i y=%i x=%i: %f\n',z,y,x,xcorr_scores(x+offsets3D(1)+1,y+offsets3D(2)+1,z+offsets3D(3)+1)));  
            end
        end
    end

  save('sample_xcorr_scores.mat','xcorr_scores');
exit
    %%
    data_cols(:,1) = reshape(chan1,[],1);
    data_cols(:,2) = reshape(chan2,[],1);
    data_cols(:,3) = reshape(chan3,[],1);
    data_cols(:,4) = reshape(chan4,[],1);

    %     %Normalize the data
    data_cols_norm = quantilenorm(data_cols);

    % reshape the normed results back into 3d images
    chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
    chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
    chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
    chan4_norm = reshape(data_cols_norm(:,4),size(chan4));

    fixed_chans_norm = (chan1_norm + chan2_norm + chan3_norm)/3;
    max_fixed = max(fixed_chans_norm,[],3);
    max_moving = max(chan4_norm,[],3);
    [corr_y,corr_x] = getXYCorrection(max_fixed,max_moving);
    
    fprintf('Round%i has a corr_y=%i and corr_x=%i\n',roundnum,corr_y,corr_x);
    
%     %These numbers were calculated using the beads dataset from Shahar 
%     %given to Dan in January 2017
%     corr_y = 2;
%     corr_x = 4;
%     
%     %make the corrected Chan4 dataset
%     chan4_corr = zeros(size(chan4));
%     %Loop over every z index of the stack
%     for z = 1:size(chan4,3)
%         chan4_corr(:,:,z) = imtranslate(squeeze(chan4(:,:,z)),[corr_x,corr_y]);
%     end
%     
%     %Save the fourth channel as a corrected
%     save3DTif(chan4_corr,sprintf('/om/project/boyden/%s/input/%s_round%i_ch03corr.tif',FOLDER_NAME,FILEROOT_NAME, roundnum));
    
% end

