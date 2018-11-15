loadParameters;

if ~params.DO_DOWNSAMPLE
    fprintf('Skipping downsample because the parameter file says not necessary\n');
    fprintf('[DONE]\n');
    return;
end

OUTPUTDIR = params.colorCorrectionImagesDir;
params = params;
%clear all;
for rnd_indx = 1:params.NUM_ROUNDS
    
    chan1_inname = fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch00.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT));
    
    if ~exist(chan1_inname,'file')
        fprintf('Skipping missing round %i \n',rnd_indx);
        continue
    end
    
    filename_colorShifts = fullfile(OUTPUTDIR,sprintf('%s-downsample_round%.03i_colorcalcs.mat',params.FILE_BASENAME,rnd_indx));
    if ~exist(filename_colorShifts,'file')
        fprintf('Skipping missing shifts file %s \n',filename_colorShifts);
        continue;
    end 
   
    chan4_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch03SHIFT.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT));
    if exist(chan4_outname,'file')
        fprintf('Skipping round which has already been done %i \n',rnd_indx);
        continue;
    end    
    load(filename_colorShifts);    
    
    %Create the symlink of chan1 to the new directory
    chan1_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch00.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT));
    command = sprintf('ln -sf %s %s',chan1_inname,chan1_outname);
    system(command);
    fprintf('Created symlink %s \n',chan1_outname);
    
    chan2 = load3DImage_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch01.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT)));
    chan2_shift = imtranslate3D(chan2,round(chan2_offsets*params.DOWNSAMPLE_RATE));
    chan2_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch01SHIFT.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT));
    save3DImage_uint16(chan2_shift,chan2_outname);
    
    chan3 = load3DImage_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch02.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT)));
    chan3_shift = imtranslate3D(chan3,round(chan3_offsets*params.DOWNSAMPLE_RATE));
    chan3_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_ch02SHIFT.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT));
    save3DImage_uint16(chan3_shift,chan3_outname);
    
    chan4 = load3DImage_uint16(fullfile(params.deconvolutionImagesDir,sprintf('%s_round%.03i_ch03.%s',params.FILE_BASENAME,rnd_indx,params.IMAGE_EXT)));
    chan4_shift = imtranslate3D(chan4,real(round(chan4_offsets*params.DOWNSAMPLE_RATE)));
    save3DImage_uint16(chan4_shift,chan4_outname);
    
    
end


postcheck = true;
for rnd_indx = 1:params.NUM_ROUNDS
    for c = 1:params.NUM_CHANNELS
        chan_outname = fullfile(params.colorCorrectionImagesDir,...
        sprintf('%s_round%.03i_%s.%s',params.FILE_BASENAME,rnd_indx,params.CHAN_STRS{c},params.IMAGE_EXT));
        if ~exist(chan_outname)
            postcheck = false;
            fprintf('[ERROR] no %s file',params.CHAN_STRS{c});
        end
    end
end

if postcheck
    fprintf('[DONE]\n');
end


