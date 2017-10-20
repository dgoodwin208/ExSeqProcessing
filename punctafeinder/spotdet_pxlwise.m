% DoG then threshold then zscore then use max of stuff then watershed. 
%% get imgs
% change RUN_NUM to not overwrite files..
RUN_NUM = 1; 
filedir = 'C:\Users\chenf\Dropbox (MIT)\Boyden Lab 2013\Broad\08_Projects\03_ATAC\Code\Spot Detection\DATA\ExSeq\No D No Up\';

ns = {'C1-PrimerN_P_frame7-1', 'C2-PrimerN_P_frame7-1', 'C3-PrimerN_P_frame7-1', 'C4-PrimerN_P_frame7-1'}; 
% cell array of names of files to be transformed w/ 2->1 transf
filepaths = cellfun(@(n) [filedir n '.tif'], ns, 'UniformOutput', false); 
outpaths  = cellfun(@(n) [n '_dis.tif'], ns, 'UniformOutput', false); 

% Z_START = 14;
% Z_END = 22; 
% NUM_Z_SLICES = Z_END - Z_START + 1; 
NUM_Z_SLICES = 42; 
% read 1st image to get single slice dimensions. [not rly necessary]
temp_img = imread(filepaths{1}, 1); 
% get img dimensions (assum. they're all the same)
img_size = size(temp_img); 

ao = zeros([img_size NUM_Z_SLICES]); %o for original
bo = zeros([img_size NUM_Z_SLICES]); 
co = zeros([img_size NUM_Z_SLICES]); 
do = zeros([img_size NUM_Z_SLICES]); 

for z = 1:NUM_Z_SLICES 
    z_ = z; 
%     z_ = Z_START + z - 1; 
    % read in z stack
    ao(:, :, z) = imread(filepaths{1}, z_);
    bo(:, :, z) = imread(filepaths{2}, z_); 
    co(:, :, z) = imread(filepaths{3}, z_); 
    do(:, :, z) = imread(filepaths{4}, z_);
end
fprintf('imgs loaded \n')
%% reference bkgd image aka 1st slice
%b1 = imread(filepaths{2}, 1); 
%c1 = imread(filepaths{3}, 1); 
b1 = min(bo,[],3);
c1 = min(co,[],3);
d1 = min(do,[],3);
a1 = min(ao,[],3);
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


adog = dog_filter(ao); 
a1dog = dog_filter2d(a1); a1dogmax = max(a1dog(:)); 
a_fgnd_mask = zeros(size(adog)); 
a_fgnd_mask(adog>a1dogmax) = 1; % above threshold
a_fgnd_mask = logical(a_fgnd_mask); 


ddog = dog_filter(do); 
d1dog = dog_filter2d(d1); d1dogmax = max(d1dog(:)); 
d_fgnd_mask = zeros(size(ddog)); 
d_fgnd_mask(ddog>d1dogmax) = 1; % above threshold
d_fgnd_mask = logical(d_fgnd_mask); 
%% Z score on Io with bkgd as ID'd by dog
bz = -Inf(size(bdog)); 
bz(b_fgnd_mask) = zscore(bo(b_fgnd_mask)); % zscore w/o bkgnd
cz = -Inf(size(cdog)); 
cz(c_fgnd_mask) = zscore(co(c_fgnd_mask)); 

az = -Inf(size(adog)); 
az(a_fgnd_mask) = zscore(ao(a_fgnd_mask)); 

dz = -Inf(size(ddog)); 
dz(d_fgnd_mask) = zscore(do(d_fgnd_mask)); 
%% max project normalized stuff; after setting bkgd to 0

% max project
%[bm, cm] = maxprojmask(bz, cz); % bm = 1 where bz >= cz
[am, bm, cm, dm] = maxprojmask(az, bz, cz, dz);
%% watershed maxprojmask(dog filtered img)
bt = bdog; 
bt(~b_fgnd_mask) = 0; % thresholded using dog background
%bt(~bm) = 0; % set non-largest regions to 0 too
masked_image = bt; 
% Q: what if not convert to int32?[answer: its kinda the same] 
neg_masked_image = -int32(masked_image); 
neg_masked_image(~masked_image) = inf; 
bL = watershed(neg_masked_image);
bL(~masked_image) = 0;
fprintf('wshed\n'); 
%%
ct = cdog; 
ct(~c_fgnd_mask) = 0; 
%ct(~cm) = 0; 
masked_image = ct; 
neg_masked_image = -int32(masked_image); 
neg_masked_image(~masked_image) = inf; 
cL = watershed(neg_masked_image);
cL(~masked_image) = 0;
fprintf('wshed\n'); 

at = adog; 
at(~a_fgnd_mask) = 0; 
%at(~am) = 0; 
masked_image = at; 
neg_masked_image = -int32(masked_image); 
neg_masked_image(~masked_image) = inf; 
aL = watershed(neg_masked_image);
aL(~masked_image) = 0;
fprintf('wshed\n');

dt = ddog; 
dt(~d_fgnd_mask) = 0; 
%dt(~dm) = 0; 
masked_image = dt; 
neg_masked_image = -int32(masked_image); 
neg_masked_image(~masked_image) = inf; 
dL = watershed(neg_masked_image);
dL(~masked_image) = 0;
fprintf('wshed\n');

%% look at results
save_img(bL, sprintf('noup%02d_bL_pipe2b.tif', RUN_NUM));  
save_img(cL, sprintf('noup%02d_cL_pipe2b.tif', RUN_NUM));  
save_img(aL, sprintf('noup%02d_aL_pipe2b.tif', RUN_NUM));  
save_img(dL, sprintf('noup%02d_dL_pipe2b.tif', RUN_NUM)); 

% results: well if two regions overlaps it just splits them into the parts
% of the venn diagram instead of eating up the smaller one completely. which
% isn't cool
