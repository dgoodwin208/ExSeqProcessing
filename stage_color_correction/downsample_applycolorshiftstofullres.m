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

%params = params;
%clear all;
parfor rnd_indx = 1:params.NUM_ROUNDS
    
    chan1_inname = fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{1},IMAGE_EXT));
    
    if ~exist(chan1_inname,'file')
        fprintf('Skipping missing round %i \n',rnd_indx);
        continue
    end
    
    filename_colorShifts = sprintf('./%s-downsample_round%.03i_colorcalcs.mat',FILE_BASENAME,rnd_indx);
    if ~exist(filename_colorShifts,'file')
        fprintf('Skipping missing shifts file %s \n',filename_colorShifts);
        continue;
    end 
   
    chan4_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{4},IMAGE_EXT);
    if exist(chan4_outname,'file')
        fprintf('Skipping round which has already been done %i \n',rnd_indx);
        continue;
    end    
    S = load(filename_colorShifts);    
    %Note: this block of code has been added since we the color correction calculation was switched
    %from bead-specific code (which required a lengthy quantilenorm) to a faster for loop, so we 
    %unpack the output of the for loop here. If this works it can be cleaned up later. DG 2019-07-20
    chan2_offsets = S.chan_offsets(2,:); 
    chan3_offsets = S.chan_offsets(3,:);
    chan4_offsets = S.chan_offsets(4,:);
    
    %Create the symlink of chan1 to the new directory
    chan1_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{1},IMAGE_EXT);
    command = sprintf('ln -sf %s %s',chan1_inname,chan1_outname);
    system(command);
    fprintf('Created symlink %s \n',chan1_outname);
    
    chan2 = load3DImage_uint16(fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{2},IMAGE_EXT)));
    chan2_shift = imtranslate3D(chan2,round(chan2_offsets*DOWNSAMPLE_RATE));
    chan2 = [];
    chan2_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{2},IMAGE_EXT);
    save3DImage_uint16(chan2_shift,chan2_outname);
    chan2_shift = [];
    
    chan3 = load3DImage_uint16(fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{3},IMAGE_EXT)));
    chan3_shift = imtranslate3D(chan3,round(chan3_offsets*DOWNSAMPLE_RATE));
    chan3 = [];
    chan3_outname = sprintf('./%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{3},IMAGE_EXT);
    save3DImage_uint16(chan3_shift,chan3_outname);
    chan3_shift = [];
    
    chan4 = load3DImage_uint16(fullfile(src_dir,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{4},IMAGE_EXT)));
    chan4_shift = imtranslate3D(chan4,real(round(chan4_offsets*DOWNSAMPLE_RATE)));
    chan4 = [];
    save3DImage_uint16(chan4_shift,chan4_outname);
    chan4_shift = [];
    
    
end
toc(t_downsample_apply);

cd(old_dir);

