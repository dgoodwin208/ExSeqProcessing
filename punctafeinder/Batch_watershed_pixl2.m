
loadParameters;
params.registeredImagesDir =  '/Users/Goody/Neuro/ExSeq/exseq20170524/4_registration-cropped';
params.transcriptResultsDir = '/Users/Goody/Neuro/ExSeq/exseq20170524/6_transcripts';
params.punctaSubvolumeDir =   '/Users/Goody/Neuro/ExSeq/exseq20170524/6_transcripts';
% filedir = 'C:\Users\chenf\Desktop\Output\Renamed\';
%filedir = '/Users/Shirin/Desktop/color/';

% File name convenction exframe7upscaled_round001_ch01
FILEROOT_NAME_INPUT = 'exseqautoframe7crop';


%parpool(8)



%M is the mask
m = {};
%centroids are the location
centroids = {};

chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};
for round_num = 1:params.NUM_ROUNDS
     
%     filepath = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_regis',FILEROOT_NAME_INPUT,round_num))
%     filepath_split2 = '_sub_B_registered.tif';
    
    for chan_num = 1:params.NUM_CHANNELS
        tic
        chan_str = chan_strs{chan_num};
        filename_in = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_registered.tif',FILEROOT_NAME_INPUT,round_num,chan_str));
        stack_in = load3DTif_uint16(filename_in);
        
        %Todo: trim the registration (not relevant in the crop)
        background = min(stack_in,[],3);
        toc
        tic
        stack_original = stack_in;
        stack_in = dog_filter(stack_in); 
  
        %min project of 3D image
        back_dog = dog_filter2d(background); 
        %avoding registration artifacts
        %2* is a magic number that just works
        back_dogmax = 2*max(max(back_dog(5:end-5,5:end-5,:))); % avoid weird edge effects
        
        fgnd_mask = zeros(size(stack_in)); 
        fgnd_mask(stack_in>back_dogmax) = 1; % use first slice to determine threshold for dog
        fgnd_mask = logical(fgnd_mask); % and get mask
        
        stack_in(~fgnd_mask) = 0; % thresholded using dog background
        
        
        %max project pxls
        %z = -Inf(size(stack_in)); 
        %z(fgnd_mask) = zscore(single(stack_original(fgnd_mask))); 
        fgnd_cell{chan_num} = fgnd_mask;
        stack_cell{chan_num} = stack_in;
        %z_cell{chan_num} = z;
       %% max project normalized stuff; after setting bkgd to 0
    end
    
    %logical OR all foregrounds together
    allmask = fgnd_cell{1} | fgnd_cell{2} | fgnd_cell{3} | fgnd_cell{4};
    
    %initializig the array of size of the 3d image
    z_cell{1} = -Inf(size(stack_in)); 
    z_cell{2} = -Inf(size(stack_in)); 
    z_cell{3} = -Inf(size(stack_in)); 
    z_cell{4} = -Inf(size(stack_in)); 
    
    %calculate the zscore of all the foreground pixels (done across channels), 
    %done per channel
    z_cell{1}(allmask) = zscore(single(stack_cell{1}(allmask))); 
    z_cell{2}(allmask) = zscore(single(stack_cell{2}(allmask))); 
    z_cell{3}(allmask) = zscore(single(stack_cell{3}(allmask))); 
    z_cell{4}(allmask) = zscore(single(stack_cell{4}(allmask))); 
    
    %re-masking foreground, now used per channel. 
    z_cell{1}(~fgnd_cell{1}) = -Inf;
    z_cell{2}(~fgnd_cell{2}) = -Inf;
    z_cell{3}(~fgnd_cell{3}) = -Inf;
    z_cell{4}(~fgnd_cell{4}) = -Inf;
    
    %Create a new mask per channel based on when a channel is the winner
    [m{1},m{2},m{3},m{4}] = maxprojmask(z_cell{1}, z_cell{2}, z_cell{3}, z_cell{4});   
   
    base_calls = zeros(size(stack_in));
    
    for chan_num = 1:4
        
        stack_in = stack_cell{chan_num};
        
        % max project
    
        %set nonlargest to 0
        stack_in(~m{chan_num}) = 0; 
        neg_masked_image = -int32(stack_in); 
        neg_masked_image(~stack_in) = inf; 
        toc
        tic
        L = uint32(watershed(neg_masked_image));
        L(~stack_in) = 0;
        fprintf('wshed\n'); 
        %watershed_out{chan_num} = L;     
        toc
        stats = regionprops(L, 'PixelIdxList', 'Area');
        centroids{round_num, chan_num} = regionprops(L, 'Centroid', 'PixelIdxList');
                
        % need to move this up before the save. 
        %eliminate spots with 10 or less voxels
        for i= 1:length(stats)
            if stats(i).Area < 10
                L(stats(i).PixelIdxList) = 0;
            else
                base_calls(stats(i).PixelIdxList) = chan_num;
            end
        end
        
        filename_out = fullfile(params.registeredImagesDir,sprintf('%s_round%.03i_%s_L.tif',FILEROOT_NAME_INPUT,round_num,chan_strs{chan_num}));
        
        save3DTif_uint16(L,filename_out);  
        %labelmatrixes{round_num, chan_num} = L;
    end
        basecalls_rounds{round_num}= base_calls;
        
end
%% 
%combine the centroid objects into a single set of centroids+pixels per
%round

puncta_centroids = cell(params.NUM_ROUNDS,1);
puncta_voxels = cell(params.NUM_ROUNDS,1);

for rnd_idx = 1:params.NUM_ROUNDS
    num_puncta_per_round = 0;
    for c_idx = 1:params.NUM_CHANNELS
        num_puncta_per_round = num_puncta_per_round + numel(centroids{rnd_idx,c_idx});
    end
    
    %initialize the vectors for the particular round
    centroids_per_round = zeros(num_puncta_per_round,3);
    voxels_per_round = cell(num_puncta_per_round,1);
    
    ctr = 1;
    for c_idx = 1:params.NUM_CHANNELS
        round_objects = centroids{rnd_idx,c_idx};
        for r_idx = 1:size(round_objects,1)
            centroids_per_round(ctr,:) = round_objects(r_idx).Centroid;
            voxels_per_round{ctr} = round_objects(r_idx).PixelIdxList;
            ctr = ctr +1;
        end
    end
    
    puncta_centroids{rnd_idx} = centroids_per_round;
    puncta_voxels{rnd_idx} = voxels_per_round;
end


filename_centroids = fullfile(params.punctaSubvolumeDir,sprintf('%s_centroids+pixels.mat',FILEROOT_NAME_INPUT));
save(filename_centroids, 'puncta_centroids','puncta_voxels', '-v7.3');

%%

basestring    = zeros(size(basecalls_rounds{1}));

for round_num = 4:20
    arraysize = size(basecalls_rounds{round_num});
    round_num
    for x=1:arraysize(1)
        for y=1:arraysize(2)
            for z=1:arraysize(3)
                if basecalls_rounds{round_num}(x,y,z) ~= 0
                    basestring(x,y,z) = basecalls_rounds{round_num}(x,y,z)*(10^(round_num-4)) + basestring(x,y,z);
               
                else
                    basestring(x,y,z) = -inf;
                end
            end
        end
    end

end


k = find(basestring>0);
blanks = find(basestring ==-inf);
basestring(blanks) = 0;
CC = bwconncomp(basestring,6);
stats = regionprops(CC,'PixelIdxList');

one = zeros(17,1);
two = zeros(17,1);
three = zeros(17,1);
four = zeros(17,1);
for i = 1:length(stats)
        strings = basestring(stats(i).PixelIdxList);
        
        temp = num2str(strings(1), '%.0f');
        for k = 1:length(temp)
            switch temp(k)
                case '1'
                    one(k,1) = one(k,1) +1;
                case '2'
                    two(k,1) = two(k,1) +1;
                case '3'
                    three(k,1) = three(k,1) +1;
                case '4'
                    four(k,1) = four(k,1) +1;
            end
        end
end

L = labelmatrix(CC);
% save_img(L, [filepath_split1 chan '_basecalls.tif']);  

% basestring_string = string(basestring,'%.0f');

plot([1:17], one./(one + two + three + four),[1:17],two./(one + two + three + four), [1:17],three./(one + two + three + four), [1:17], four./(one + two + three + four));
%%
%for i = 1:16
%    for j = 1: length(k)
%        
%    end
%end

    
%%merge frame 75
for chan_num = 1:4
    for round_num = 1:20
         filepath_split1 = fullfile(filedir,sprintf('%s_round%.03i',FILEROOT_NAME_INPUT,round_num))
        filepath_split2 = '_sub_B_registered.tif';
    
        
        if chan_num == 1
           chan = sprintf('_ch0%i',chan_num-1)
        else
            chan = sprintf('_ch0%iSHIFT',chan_num-1);
        end
        stack_in = uint16(load3DTif([filepath_split1 chan  filepath_split2]));
        extractedframe(:,:,round_num) = stack_in(:,:,75);
    end
    save_img(extractedframe, [filepath_split1 chan '_raw75' filepath_split2]);
end    


%% calculating overalps with nearest neighbor

centallchannels = {};

for round_num = 4:19
    allchannels = [];
    centcall = [];
    for chan_num = 1:4
        for i = 1:length(centroids{round_num, chan_num})
            temp = centroids{round_num, chan_num}(i).Centroid;
            allchannels = [allchannels;  temp];
            centcall = [centcall; chan_num];
        end
    end
    centallchannels{round_num} = allchannels;
    centcall_rounds{round_num} = centcall;
end

    basecallclosest = strings(20,length(centallchannels{4}));
    for round_num = 5:19
        T = delaunayn(centallchannels{round_num});
        [k,d] = dsearchn(centallchannels{round_num}, T,centallchannels{4}); %finds in the indexes of 5 whihc are closest to points in 4
        %find indexes whihc are >2 std away from mean of distance
        faridx = find(d>(mean(d) +2*std(d)));
        for i = 1:length(k)
            if find(faridx==i)
               basecallclosest(round_num,i) = '0';
            else               
               basecallclosest(round_num,i) = num2str(centcall_rounds{round_num}(k(i)));
            end
        end
        
    end
    for j = 1:length(centallchannels{4})
         basecallclosest(4,j) = num2str(centcall_rounds{4}(j));
    end