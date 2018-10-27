% DoG then threshold then zscore 
% but also watershed and then use max zscore stuff on watershed components
%% get imgs
% change RUN_NUM to not overwrite files..
RUN_NUM = 321; 
filedir = 'C:\Users\Maryann\Desktop\leBoyden\imgRegistration\DATA\';

ns = {'C1-PL(L)', 'C2-PL(L)', 'C3-PL(L)', 'C4-PL(L)'}; 
% cell array of names of files to be transformed w/ 2->1 transf
filepaths = cellfun(@(n) [filedir n '.tif'], ns, 'UniformOutput', false); 
outpaths  = cellfun(@(n) [n '_dis.tif'], ns, 'UniformOutput', false); 

% Z_START = 17;
% Z_END = 21; 
% NUM_Z_SLICES = Z_END - Z_START + 1; 
NUM_Z_SLICES = 69; 
% read 1st image to get single slice dimensions. [not rly necessary]
temp_img = imread(filepaths{1}, 1); 
% get img dimensions (assum. they're all the same)
img_size = size(temp_img); 

%ao = zeros([img_size NUM_Z_SLICES]); %o for original
bo = zeros([img_size NUM_Z_SLICES]); 
co = zeros([img_size NUM_Z_SLICES]); 
%do = zeros([img_size NUM_Z_SLICES]); 

for z = 1:NUM_Z_SLICES 
    z_ = z; 
%     z_ = Z_START + z - 1; 
    % read in z stack
    %ao(:, :, z) = imread(filepaths{1}, z_);
    bo(:, :, z) = imread(filepaths{2}, z_); 
    co(:, :, z) = imread(filepaths{3}, z_); 
    %do(:, :, z) = imread(filepaths{4}, z_);
end
fprintf('imgs loaded \n')
%% reference bkgd image aka 1st slice
b1 = imread(filepaths{2}, 1); 
c1 = imread(filepaths{3}, 1); 
%% DoG 
%TODO change parameters in dog as needed
bdog = dog_filter(bo); 
b1dog = dog_filter2d(b1); b1dogmax = max(b1dog(:)); 
b_fgnd_mask = zeros(size(bdog)); 
b_fgnd_mask(bdog>b1dogmax) = 1; % use first slice to determine threshold for dog
b_fgnd_mask = logical(b_fgnd_mask); % and get mask
cdog = dog_filter(co); 
c1dog = dog_filter2d(c1); c1dogmax = max(c1dog(:)); 
c_fgnd_mask = zeros(size(cdog)); 
c_fgnd_mask(cdog>c1dogmax) = 1; % above threshold
c_fgnd_mask = logical(c_fgnd_mask); 
%% Z score on Io with bkgd as ID'd by dog
bn = -Inf(size(bdog)); 
bn(b_fgnd_mask) = zscore(bo(b_fgnd_mask)); % zscore w/o bkgnd
cn = -Inf(size(cdog)); 
cn(c_fgnd_mask) = zscore(co(c_fgnd_mask)); 

%% watershed maxprojmask(dog filtered img)
bt = bdog; 
bt(~b_fgnd_mask) = 0; % thresholded using dog background
masked_image = bt; 
% TODO: what if not convert to int32?[answer: its kinda the same] 
neg_masked_image = -int32(masked_image); 
neg_masked_image(~masked_image) = inf; 
bL = watershed(neg_masked_image);
bL(~masked_image) = 0;
fprintf('wshed\n'); 
stats = regionprops(bL, 'PixelIdxList', 'Area');
for i= 1:length(stats) % also try bwareaopen
    if stats(i).Area < 6
        bL(stats(i).PixelIdxList) = 0;
    end
end
%%
ct = cdog; 
ct(~c_fgnd_mask) = 0; % thresholded using dog background
masked_image = ct; 
% TODO: what if not convert to int32?[answer: its kinda the same] 
neg_masked_image = -int32(masked_image); 
neg_masked_image(~masked_image) = inf; 
cL = watershed(neg_masked_image);
cL(~masked_image) = 0;
fprintf('wshed\n'); 
stats = regionprops(cL, 'PixelIdxList', 'Area');
for i= 1:length(stats) % also try bwareaopen
    if stats(i).Area < 6
        cL(stats(i).PixelIdxList) = 0;
    end
end
%% get intersections
interx = bL & cL; % TODO: eventually need pairwise?
cc = bwconncomp(interx); % connected components
stats = regionprops(cc, 'PixelIdxList'); % (disjoint sets)
%%
% compare regions 
tic
for k = 1:length(stats) % SO SLOW.
    pxls = stats(k).PixelIdxList; % array of lin ind
        if mean(bn(pxls)) > mean(cn(pxls)) % mean of normalized [uh oh -Inf]
            % keep b and 
            % find original labeled component in c, set to 0
            pxlmatches = (cL == cL(pxls(1))); 
            cL(pxlmatches) = 0; % they all have the same #
        else
            % set everyone in b to zero
            pxlmatches = (bL == bL(pxls(1))); 
            bL(pxlmatches) = 0;
        end
        if mod(k, 100) == 0
            fprintf('k: %d\n', k) 
            toc
            tic
        end
end
% after this all xL's should be disjoint.. [check nnz(bl&cL)]
toc

%% look at results
save_img(bL, sprintf('trial%02d_bL_pipe2b.tif', RUN_NUM));  
save_img(cL, sprintf('trial%02d_cL_pipe2b.tif', RUN_NUM));  
% results: well if two regions overlaps it just splits them into the parts
% of the venn diagram instead of eating up the smaller one completely. which
% isn't cool
