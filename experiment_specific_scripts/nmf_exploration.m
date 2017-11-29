%% Exploring NMF on the splintr data

img = load3DTif_uint16('/Users/Goody/Neuro/ExSeq/exseq20170524/exseqautoframe7_3_round007_ch02SHIFT_registered.tif');

img_patch = img(101:150, 101:150,:);
%%

X = zeros(size(img_patch,1)*size(img_patch,2),size(img_patch,3));

for z_idx = 1:size(X,2)    
    X(:,z_idx) = reshape(squeeze(img_patch(:,:,z_idx)),[],1);    
end
%%
k=50;
tic
option.eta=max(img_patch(:))^3;
[A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(X,k);
toc
%%
figure; 
for idx = 1:k
    t = A(:,idx)*Y(idx,:);
    subplot(1,2,1);
    plot(Y(idx,:));
    s = reshape(t,50,50,141);
    subplot(1,2,2)
    imagesc(max(s,[],3))
    title(sprintf('Component %i',idx));
    pause
end


%% What does the low-rank approximation of the raw data look like?
%incomplet
% X_approx = A*Y;
%
% pixel_img = zeros(params.NUM_ROUNDS*PSIZE,params.NUM_CHANNELS*PSIZE);
%
% last_puncta_idx = 0;
% puncta_vol = zeros(PSIZE,PSIZE,PSIZE);
%
% puncta_idx = 0;
% for pixel_index = 1:10000
%
%     puncta_idx = floor(pixel_index/(PSIZE*PSIZE*PSIZE))+1;
%
%     if puncta_idx>last_puncta_idx
%         if last_puncta_idx~=0
%             figure;
%             subplot(1,2,1)
%             imagesc(max(puncta_vol,[],3))
%             title(sprintf('Puncta call for puncta idx %i',puncta_idx))
%             for r_idx = 1:params.NUM_ROUNDS
%                 for c_idx = 1:params.NUM_CHANNELS
%                     pixel_img((r_idx-1)*PSIZE+1:r_idx*PSIZE,(c_idx-1)*PSIZE+1:c_idx*PSIZE)=...
%                         max(squeeze(puncta_set(:,:,:,r_idx,c_idx,puncta_idx)),[],3);
%                 end
%             end
%             subplot(1,2,2)
%             imagesc(pixel_img)
%             title(sprintf('Puncta call for puncta idx %i',puncta_idx))
%
%         end
%
%
%         puncta_vol = zeros(PSIZE,PSIZE,PSIZE);
%         last_puncta_idx = puncta_idx;
%     end
%
%     positioning_index = mod(pixel_index-1,(PSIZE*PSIZE*PSIZE))+1;
%     [x_idx,y_idx,z_idx] = ind2sub([PSIZE,PSIZE,PSIZE],positioning_index);
%     puncta_vol(x_idx,y_idx,z_idx) = calls(pixel_index);
% end



%% Load the ground truth codes and turn them into 1D vectors

%Linearize the colors into a vector, ordered by color, then round. Example:
%[base1_color1,base1_color2,base1_color3,base1_color4,base2_color1,...]
%Add the extra +1 to create an element of all zeros
known_dictionary = zeros(params.NUM_ROUNDS*params.NUM_CHANNELS,size(groundtruth_codes,1)+1);

%The ground truth codes are saved in a row
for code_idx = 1:size(groundtruth_codes,1)
    for rnd_idx = 1:params.NUM_ROUNDS
        offset = (rnd_idx-1)*params.NUM_CHANNELS;
        known_dictionary(offset+groundtruth_codes(code_idx,rnd_idx),code_idx)=1;
    end
end

%% Transform punctaset into a set of pixel histories

%Make a history of each pixel: [length(pixel histry) x number of puncta]
pixel_histories = zeros(params.NUM_ROUNDS*params.NUM_CHANNELS,...
    size(puncta_set,6)*PSIZE*PSIZE*PSIZE);

for p_idx = 1:size(puncta_set,6)
    
    offset_ctr =1;
    for x_idx = 1:PSIZE
        for y_idx = 1:PSIZE
            for z_idx = 1:PSIZE
                
                %Puncta array should be size[experiments,colors]
                puncta_array = squeeze(puncta_set(x_idx,y_idx,z_idx,:,:,p_idx));
                
                %So if we take the transpose and then reshape, it should
                %put into the desired order
                puncta_history = reshape(puncta_array',1,[]);
                
                %calculate it's position in puncta_histories
                puncta_idx = (p_idx-1)*1000+offset_ctr;
                
                pixel_histories(:,puncta_idx) = puncta_history;
                offset_ctr = offset_ctr+1;
                
            end
        end
    end
    
    if mod(p_idx,100)==0
        fprintf('Processed %i/%i\n',p_idx, size(X,2))
    end
end

%% Try NMF without a per-pixel basis

k=30;
tic
[A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(pixel_histories,k);
toc


%% Do a least squares with L1 penalty

X = pixel_histories;
A = known_dictionary;

k = size(known_dictionary,2);

option.beta = .1;

Ae=[A;sqrt(option.beta)*ones(1,k)];
X0=[X;zeros(1,size(X,2))];
Y=kfcnnls(Ae,X0);

%% Look at his
[vals,calls] = max(Y,[],1);
figure;
[counts,edges]=histcounts(calls,k);
bar(counts);

gtlabels = {};
for i = 1:size(groundtruth_codes,1)
    transcript_string = '';
    for c = 1:size(groundtruth_codes,2)
        transcript_string(c) = num2str(groundtruth_codes(i,c));
    end
    gtlabels{i}=transcript_string;
end
gtlabels{k}='background';

xticklabel_rotate(1:k,45,gtlabels,'interpreter','none')

%Let's visualize these pixel calls

pixel_img = zeros(params.NUM_ROUNDS*PSIZE,params.NUM_CHANNELS*PSIZE);

last_puncta_idx = 0;
puncta_vol = zeros(PSIZE,PSIZE,PSIZE);

for pixel_index = 1:10000
    
    puncta_idx = floor(pixel_index/(PSIZE*PSIZE*PSIZE))+1;
    
    if puncta_idx>last_puncta_idx
        if last_puncta_idx~=0
            figure;
            subplot(1,2,1)
            %             imagesc(max(puncta_vol,[],3))
            imagesc(puncta_vol(:,:,5))
            title(sprintf('Puncta call for puncta idx %i',puncta_idx))
            
            
            for r_idx = 1:params.NUM_ROUNDS
                for c_idx = 1:params.NUM_CHANNELS
                    pixel_img((r_idx-1)*PSIZE+1:r_idx*PSIZE,(c_idx-1)*PSIZE+1:c_idx*PSIZE)=...
                        max(squeeze(puncta_set(:,:,:,r_idx,c_idx,puncta_idx)),[],3);
                end
            end
            subplot(1,2,2)
            imagesc(pixel_img)
            title(sprintf('Puncta call for puncta idx %i',puncta_idx))
            
        end
        
        
        puncta_vol = zeros(PSIZE,PSIZE,PSIZE);
        last_puncta_idx = puncta_idx;
    end
    
    positioning_index = mod(pixel_index-1,(PSIZE*PSIZE*PSIZE))+1;
    [x_idx,y_idx,z_idx] = ind2sub([PSIZE,PSIZE,PSIZE],positioning_index);
    puncta_vol(x_idx,y_idx,z_idx) = calls(pixel_index);
end

%% What if we did a rank-2 NMF per puncta subvolume, and share a background model?

%Let's just try
for p_idx = indices_to_view %1:10:1000
    
    puncta = puncta_set(:,:,:,:,:,p_idx);
    
    
    
%     %sum of channels
%     puncta_roundref = sum(squeeze(puncta(:,:,:,params.REFERENCE_ROUND_PUNCTA,:)),4);
%     offsetrange = [2,2,2];
%     %Assuming the first round is reference round for puncta finding
%     %Todo: take this out of prototype
%     moving_exp_indices = 1:params.NUM_ROUNDS; moving_exp_indices(params.REFERENCE_ROUND_PUNCTA) = [];
%     for e_idx = moving_exp_indices 
%         %Get the sum of the colors for the moving channel
%         puncta_roundmove = sum(squeeze(puncta(:,:,:,e_idx,:)),4);
%         [~,shifts] = crossCorr3D(puncta_roundref,puncta_roundmove,offsetrange);
%         for c_idx = 1:params.NUM_CHANNELS
%             puncta(:,:,:,e_idx,c_idx) = ...
%                 imtranslate3D(squeeze(puncta(:,:,:,e_idx,c_idx)),shifts);
%         end
%     end
%     
%     makeImageOfPunctaROIs(puncta,params,1)
    
    
    pix_ctr = 1; colexp_ctr = 1;
    puncta_full_img = zeros(params.NUM_ROUNDS*PSIZE,params.NUM_CHANNELS*PSIZE*PSIZE);
    for x_idx = 1:PSIZE
        for y_idx = 1:PSIZE
            for z_idx = 1:PSIZE
                
                %Looping explicitly over color and experiment until I get
                %reshape doing what we expect
                colexp_ctr = 1;
                for e_idx = 1:params.NUM_ROUNDS
                    for c_idx = 1:params.NUM_CHANNELS
                        val = puncta_set(x_idx,y_idx,z_idx,e_idx,c_idx,p_idx);
                        pixel_histories(pix_ctr,colexp_ctr) = val;
                        colexp_ctr = colexp_ctr+1;
                        
                        
                        y_loc = (z_idx-1)*PSIZE*params.NUM_CHANNELS + (c_idx-1)*PSIZE+y_idx;
                        
                        puncta_full_img((e_idx-1)*PSIZE+x_idx,y_loc) = val;
                    end
                end
                pix_ctr = pix_ctr+1;
                
            end
        end
    end
    figure(2); imagesc(puncta_full_img);
    
    
    
    
    [A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(pixel_histories,2);
    
    mask1 = zeros(PSIZE,PSIZE,PSIZE);
    mask2 = zeros(PSIZE,PSIZE,PSIZE);
    offset_ctr =1;
    for x_idx = 1:PSIZE
        for y_idx = 1:PSIZE
            for z_idx = 1:PSIZE
                
                mask1(x_idx,y_idx,z_idx) = A(offset_ctr,1);
                mask2(x_idx,y_idx,z_idx) = A(offset_ctr,2);
                
                offset_ctr = offset_ctr+1;
                
            end
        end
    end
    
    pixel_img1 = zeros(PSIZE,PSIZE*PSIZE);
    pixel_img2 = zeros(PSIZE,PSIZE*PSIZE);
    pixel_img3 = zeros(PSIZE,PSIZE*PSIZE);
    
    
    for z_idx = 1:PSIZE
        
        pixel_img1(:,(z_idx-1)*PSIZE+1:z_idx*PSIZE) = squeeze(mask1(:,:,z_idx));
        pixel_img2(:,(z_idx-1)*PSIZE+1:z_idx*PSIZE) = squeeze(mask2(:,:,z_idx));
        pixel_img3(:,(z_idx-1)*PSIZE+1:z_idx*PSIZE) = squeeze(mask1(:,:,z_idx))>squeeze(mask2(:,:,z_idx));
        
        
    end
    
    figure(3);
    subplot(3,1,1)
    imagesc(pixel_img1)
    title(sprintf('Component 1 of puncta %i',p_idx))
    
    subplot(3,1,2)
    imagesc(pixel_img2)
    title('Component 2')
    
    subplot(3,1,3)
    imagesc(pixel_img3)
    title('Mask')
    
    pause
end



