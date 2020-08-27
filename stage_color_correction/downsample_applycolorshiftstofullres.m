loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    fprintf('[DONE]\n');
    return;
end

t_downsample_apply = tic;
delete(gcp('nocreate'))
conditions = conditions_for_concurrency();
max_pool_size = concurrency_size_in_downsample_apply(conditions);

parpool(max_pool_size);

% for avoiding the violation of trancsparency
FILE_BASENAME = params.FILE_BASENAME;
CHAN_STRS = params.CHAN_STRS;
SHIFT_CHAN_STRS = params.SHIFT_CHAN_STRS;
IMAGE_EXT = params.IMAGE_EXT;
DOWNSAMPLE_RATE = params.DOWNSAMPLE_RATE;

src_dir = relpath(params.colorCorrectionImagesDir,params.deconvolutionImagesDir);
old_dir = pwd;
cd(params.colorCorrectionImagesDir);

parfor rnd_indx = 1:params.NUM_ROUNDS
    
    %Check to see if there is a problem with missing data
    
    chan1_inname = fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{1},IMAGE_EXT));
    if ~exist(chan1_inname,'file')
        fprintf('Skipping missing round %i \n',rnd_indx);
        continue
    end
    
    %Check to see if this round has already been processed
    chan_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{end},IMAGE_EXT);
    if exist(chan_outname,'file')
        fprintf('Skipping round which has already been done %i \n',rnd_indx);
        continue;
    end 
    
    filename_colorShifts = sprintf('./%s-downsample_round%.03i_colorcalcs.mat',FILE_BASENAME,rnd_indx);
    if ~exist(filename_colorShifts,'file')
        fprintf('Skipping missing shifts file %s \n',filename_colorShifts);
        continue;
    end 
   

    %Load the computed offsets
    S = load(filename_colorShifts);
    
    
    %Create the symlink of chan1 to the new directory
    chan1_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{1},IMAGE_EXT);
    command = sprintf('ln -sf %s %s',chan1_inname,chan1_outname);
    system(command);
    fprintf('Created symlink %s \n',chan1_outname);
    
    %Note that this makes the strong assumption that the first ccolor
    %channel is the reference and will not be modified. -DG 
    for c_idx = 2:params.NUM_CHANNELS
        chan_offsets = S.chan_offsets(c_idx,:);
        
        %Load the data and apply shift
        chan = load3DImage_uint16(fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{c_idx},IMAGE_EXT)));
        chan_shift = imtranslate3D(chan,round(chan_offsets*DOWNSAMPLE_RATE));
        
        chan_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{c_idx},IMAGE_EXT);
        save3DImage_uint16(chan_shift,chan_outname);
        
    end
    chan = [];
    chan_shift = [];
    %If there is a morphology round, it typically is the same sequencing
    %chemistry, plus an additional color channel for the morphology
    if isfield(params, 'MORPHOLOGY_ROUND') && (rnd_indx == params.MORPHOLOGY_ROUND)
        chan_offsets = S.chan_offsets(params.NUM_CHANNELS+1,:);

        chan = load3DImage_uint16(fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,params.MORPHOLOGY_CHAN_STR,IMAGE_EXT)));
        chan_shift = imtranslate3D(chan,real(round(chan_offsets*DOWNSAMPLE_RATE)));
        chan = [];
        chan_outname = sprintf('./%s_round%.03i_%sSHIFT.%s',FILE_BASENAME,rnd_indx,params.MORPHOLOGY_CHAN_STR,IMAGE_EXT);
        save3DImage_uint16(chan_shift,chan_outname);
        chan_shift = [];
    end
    
    
end
toc(t_downsample_apply);

cd(old_dir);

