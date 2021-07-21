%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function calcCorrespondencesForFixed(fixed_run)

loadParameters;

if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample_',params.FILE_BASENAME);
else
    filename_root = sprintf('%s_',params.FILE_BASENAME);
end

fprintf('CalcCorrespondencesForFixed FIXED: %i\n', fixed_run);

lf_sift_filename = fullfile(params.registeredImagesDir,sprintf('%sround%03d_lf_sift.mat',filename_root,fixed_run));
if exist(lf_sift_filename)
    disp('lf_sift files are existed.');
    return;
end

filename = fullfile(params.normalizedImagesDir,sprintf('%sround%03d_%s.%s',...
    filename_root,fixed_run,regparams.CHANNELS{1},params.IMAGE_EXT ));

img_total_size = image_dimensions(filename);
ymin = 1;
ymax = img_total_size(1);
xmin = 1;
xmax = img_total_size(2);


%------------------------------Load Descriptors -------------------------%
%Load all descriptors for the FIXED channel
tic;
keys_fixed_sift.pos = [];
keys_fixed_sift.ivec = [];
for register_channel = [regparams.REGISTERCHANNELS_SIFT]
    descriptor_output_dir_fixed = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        fixed_run,register_channel{1}));

    filename = fullfile(descriptor_output_dir_fixed, ...
        [num2str(xmin) '-' num2str(xmax) '_' num2str(ymin) '-' num2str(ymax) '.mat']);

    data = load(filename);
    keys = vertcat(data.keys{:});
    pos = [[keys(:).y]',[keys(:).x]',[keys(:).z]'];
    ivec = vertcat(keys(:).ivec);

    keys_fixed_sift.pos  = vertcat(keys_fixed_sift.pos,pos);
    keys_fixed_sift.ivec = vertcat(keys_fixed_sift.ivec,ivec);
end
fprintf('load sift keys of fixed round%03d. ',fixed_run);toc;

tic;
keys_fixed_sc.pos = [];
for register_channel = [regparams.REGISTERCHANNELS_SC]
    descriptor_output_dir_fixed = fullfile(params.registeredImagesDir,sprintf('%sround%03d_%s/',filename_root, ...
        fixed_run,register_channel{1}));

    filename = fullfile(descriptor_output_dir_fixed, ...
        [num2str(xmin) '-' num2str(xmax) '_' num2str(ymin) '-' num2str(ymax) '.mat']);

    data = load(filename);
    keys = vertcat(data.keys{:});
    pos = [[keys(:).y]',[keys(:).x]',[keys(:).z]'];

    keys_fixed_sc.pos = vertcat(keys_fixed_sc.pos,pos);
end
fprintf('load sc keys of fixed round%03d. ',fixed_run);toc;
%------------All descriptors are now loaded as keys_*_total -------------%


num_keys_fixed = length(keys_fixed_sift)+length(keys_fixed_sc);
disp(['Sees ' num2str(num_keys_fixed) ' features for fixed']);
if num_keys_fixed==0
    disp('[ERROR] Empty set of descriptors.')
    return;
end

% ----------- SIFT MATCHING AND ROBUST MODEL SELECTION ----------%
%

%Extract the keypoints-only for the shape context calculation
%F for fixed
DF_SIFT = keys_fixed_sift.ivec;
LF_SIFT = keys_fixed_sift.pos;
LF_SC = keys_fixed_sc.pos;
fprintf('prepare keypoints of fixed round.');toc;


fprintf('normalizing SIFT descriptors...\n');
tic;
DF_SIFT = double(DF_SIFT);
DF_SIFT_norm = DF_SIFT ./ repmat(sum(DF_SIFT,2),1,size(DF_SIFT,2));
clear DF_SIFT;
size(DF_SIFT_norm)
toc;


% 2020-09-09 removed the ShapeContext features. These are not being used currently
% can be reintroduced easily for the case of challenging registrations. -Dan G
%fprintf('calculating ShapeContext descriptors (removed)...\n');
%tic;
%We create a shape context descriptor for the same keypoint
%that has the SIFT descriptor.
%So we calculate the SIFT descriptor on the normed channel
%(summedNorm), and we calculate the Shape Context descriptor
%using keypoints from all other channels
%DF_SC=ShapeContext(LF_SIFT,LF_SC);

%toc;

tic;
lf_sift_filename = fullfile(params.registeredImagesDir,sprintf('%sround%03d_lf_sift.mat',...
    filename_root,fixed_run));
save(lf_sift_filename,'LF_SIFT','DF_SIFT_norm','img_total_size','num_keys_fixed','-v7.3');

