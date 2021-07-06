function completionReport(yamlfile)

%Use a YAML file to load the parameters for a large experiment
yamlspecs = ReadYaml(yamlfile);
%The YAML files might have extra quotes so remove them
BASENAME = strrep(yamlspecs.basename,'''','');
NUM_FOVS = prod(yamlspecs.montage_size);
if isfield(yamlspecs,'reg_type')
    REG_TYPE = yamlspecs.reg_type;
else
    REG_TYPE = 'affine';
end
REG_ROUND = yamlspecs.ref_round;
NUM_ROUNDS = yamlspecs.rounds;
if isfield(yamlspecs,'maxnum_missing')
    MAXNUM_MISSING = yamlspecs.maxnum_missing;
else
    MAXNUM_MISSING = 2;
end
ROOT_DIR = strrep(yamlspecs.base_dir,'''','');

%Initialize the file
expResults = {};


for F = 0:NUM_FOVS
    expR= struct;
    expR.F = F;
    expR.punctafeinder_centroids = -1;
    expR.num_complete_puncta = -1;
    expR.num_rounds_puncta = -1;
    expR.percentage_volume_usable = -1;
    expResults{end+1} = expR;
end

for F = 0:NUM_FOVS
    f_idx = F+1; %because 0/1 indexing
    EXP_NAME = sprintf('%s-F%.3i',BASENAME,F);
    
    puncta_subvol_dir = ...
        fullfile(ROOT_DIR, sprintf('F%.3i/5_puncta-extraction',F));
    filename_centroids = fullfile(puncta_subvol_dir,sprintf('%s_centroids+pixels.mat',EXP_NAME));
    if exist(filename_centroids,'file')
        matObj = matfile(filename_centroids,'Writable',false);
        num_centroids = size(matObj,'puncta_centroids');
        fprintf('CHECK: F%.3i has _centroids+pixels, %i centroids discovered\n',...
            F,num_centroids(1));
        
        expResults{f_idx}.punctafeinder_centroids = num_centroids(1);
    else
        fprintf('FAIL: F%.3i does not have _centroids+pixels.\n',F);
        continue
    end
    
    filename_puncta = fullfile(puncta_subvol_dir,sprintf('%s_punctavoxels.mat',EXP_NAME ));
    
    if exist(filename_puncta,'file')
        load(filename_puncta,'puncta_indices_cell')
        num_puncta = size(puncta_indices_cell{1});
        fprintf('CHECK: F%.3i has _punctavoxels, %i puncta saved from %i rounds\n',...
            F,num_puncta(1),length(puncta_indices_cell));
        
        expResults{f_idx}.num_complete_puncta = num_puncta(1);
        expResults{f_idx}.num_rounds_puncta = length(puncta_indices_cell);
    else
        fprintf('FAIL: F%.3i does not have punctavoels.\n',F);
        continue
    end
    
    
    
    %Step two; what's the overlap of the registered volumes?
    
    registration_dir = fullfile(ROOT_DIR, sprintf('F%.3i/4_registration',F));
    try
        for rnd = 1:NUM_ROUNDS
            img_filename = fullfile(registration_dir, sprintf('%s-downsample_round%.3i_summedNorm_%s.h5',...
                EXP_NAME,rnd,REG_TYPE));
            if rnd == REG_ROUND
                norm_dir = fullfile(ROOT_DIR, sprintf('F%.3i/3_normalization',F));
                img_filename = fullfile(norm_dir, sprintf('%s-downsample_round%.3i_summedNorm.h5',...
                    EXP_NAME,rnd));
            end
            img = load3DImage_uint16(img_filename);
            
            if rnd==1 %initialize the zero counter
                zero_counter = zeros(size(img));
            end
            %the value zero is the mark of the outside the registration
            zero_counter = zero_counter + (img==0);
        end
    catch
        fprintf('FAIL: Couldnt load file %s\n',img_filename)
        continue
    end
    %Get the total volume
    unusable_pixel_mask = zero_counter>MAXNUM_MISSING;
    perc_usable_data = 1- sum(unusable_pixel_mask(:))/numel(unusable_pixel_mask);
    
    expResults{f_idx}.percentage_volume_usable = perc_usable_data ;
end

output_file = fullfile(ROOT_DIR,sprintf('compReport-%s.mat',date));
save(output_file,'expResults')

