loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    fprintf('[DONE]\n');
    return;
end

max_pool_size = concurrency_size_in_downsample_apply();

parpool(max_pool_size);

% for avoiding the violation of trancsparency
INPUTDIR = params.deconvolutionImagesDir;
OUTPUTDIR = params.colorCorrectionImagesDir;
FILE_BASENAME = params.FILE_BASENAME;
CHAN_STRS = params.CHAN_STRS;
SHIFT_CHAN_STRS = params.SHIFT_CHAN_STRS;
IMAGE_EXT = params.IMAGE_EXT;
DOWNSAMPLE_RATE = params.DOWNSAMPLE_RATE;
%params = params;
%clear all;
parfor rnd_indx = 1:params.NUM_ROUNDS
    
    chan1_inname = fullfile(INPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{1},IMAGE_EXT));
    
    if ~exist(chan1_inname,'file')
        fprintf('Skipping missing round %i \n',rnd_indx);
        continue
    end
    
    filename_colorShifts = fullfile(OUTPUTDIR,sprintf('%s-downsample_round%.03i_colorcalcs.mat',FILE_BASENAME,rnd_indx));
    if ~exist(filename_colorShifts,'file')
        fprintf('Skipping missing shifts file %s \n',filename_colorShifts);
        continue;
    end 
   
    chan4_outname = fullfile(OUTPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{4},IMAGE_EXT));
    if exist(chan4_outname,'file')
        fprintf('Skipping round which has already been done %i \n',rnd_indx);
        continue;
    end    
    S = load(filename_colorShifts);    
    
    %Create the symlink of chan1 to the new directory
    chan1_outname = fullfile(OUTPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{1},IMAGE_EXT));
    command = sprintf('ln -sf %s %s',chan1_inname,chan1_outname);
    system(command);
    fprintf('Created symlink %s \n',chan1_outname);
    
    chan2 = load3DImage_uint16(fullfile(INPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{2},IMAGE_EXT)));
    chan2_shift = imtranslate3D(chan2,round(S.chan2_offsets*DOWNSAMPLE_RATE));
    chan2 = [];
    chan2_outname = fullfile(OUTPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{2},IMAGE_EXT));
    save3DImage_uint16(chan2_shift,chan2_outname);
    chan2_shift = [];
    
    chan3 = load3DImage_uint16(fullfile(INPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{3},IMAGE_EXT)));
    chan3_shift = imtranslate3D(chan3,round(S.chan3_offsets*DOWNSAMPLE_RATE));
    chan3 = [];
    chan3_outname = fullfile(OUTPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,SHIFT_CHAN_STRS{3},IMAGE_EXT));
    save3DImage_uint16(chan3_shift,chan3_outname);
    chan3_shift = [];
    
    chan4 = load3DImage_uint16(fullfile(INPUTDIR,sprintf('%s_round%.03i_%s.%s',FILE_BASENAME,rnd_indx,CHAN_STRS{4},IMAGE_EXT)));
    chan4_shift = imtranslate3D(chan4,real(round(S.chan4_offsets*DOWNSAMPLE_RATE)));
    chan4 = [];
    save3DImage_uint16(chan4_shift,chan4_outname);
    chan4_shift = [];
    
    
end

