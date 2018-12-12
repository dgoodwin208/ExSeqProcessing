function reprocessRoundFromColorCorr(rounds)
loadParameters;
if params.DO_DOWNSAMPLE
    filename_root = sprintf('%s-downsample',params.FILE_BASENAME);
else
    filename_root = sprintf('%s',params.FILE_BASENAME);
end

%Color Correct from File
%fprintf('COLOR CORRECTION:\n');
for rnd = rounds
    colorcorrection_3D_from_file(rnd);
end;

%------ Normalization ------%
%First remove the previous files
 fprintf('NORMALIZATION:\n');
 for rnd = rounds
     outputfile= sprintf('%s/%s_round%03i_summedNorm.tif',params.normalizedImagesDir,params.FILE_BASENAME,rnd);
     delete(outputfile)
     fprintf('Deleted noramlized file: %s\n',outputfile);
     outputfile= sprintf('%s/%s-downsample_round%03i_summedNorm.tif',params.normalizedImagesDir,params.FILE_BASENAME,rnd);
     delete(outputfile)
     fprintf('Deleted downsampled noramlized file: %s\n',outputfile);
 end
% 

normalization(params.colorCorrectionImagesDir,params.normalizedImagesDir,params.FILE_BASENAME,params.SHIFT_CHAN_STRS,rounds,false);
normalization(params.colorCorrectionImagesDir,params.normalizedImagesDir,params.FILE_BASENAME,params.SHIFT_CHAN_STRS,rounds,true);

%------ Registration ------%
fprintf('REGISTRATION:\n');
%First remove the previous files
%Folders of descriptors
for rnd = rounds
    for register_channel = unique([regparams.REGISTERCHANNELS_SIFT,regparams.REGISTERCHANNELS_SC])
        regChan = register_channel{1};
        descriptor_output_dir = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s/',filename_root,rnd,regChan));
        if exist(descriptor_output_dir,'dir')
            rmcommand = sprintf('rm -rf %s',descriptor_output_dir);
            system(rmcommand);
            fprintf('%s\n',rmcommand);
        end
    end
end
%global keys files
for rnd = rounds
    output_keys_filename = fullfile(params.registeredImagesDir,sprintf('globalkeys_%s_round%03d.mat',filename_root,rnd));
    delete(output_keys_filename)
    fprintf('Deleted correspondences file: %s\n',output_keys_filename); 
end
%Warped files
for rnd = rounds
    
    output_affine_filename = fullfile(params.registeredImagesDir,sprintf('%s-downsample_round%03d_%s_affine.tif',params.FILE_BASENAME,rnd,regparams.CHANNELS{end}));
    delete(output_affine_filename)
    fprintf('Deleted warped downsampled file: %s\n',output_affine_filename); 
    output_affine_filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_affine.tif',params.FILE_BASENAME,rnd,regparams.CHANNELS{end}));
    delete(output_affine_filename)
    fprintf('Deleted warped fullres file: %s\n',output_affine_filename); 
    
    output_TPS_filename = fullfile(params.registeredImagesDir,sprintf('TPSMap_%s_round%03d.mat',params.FILE_BASENAME,rnd));
    delete(output_TPS_filename)
    fprintf('Deleted TPS file: %s\n',output_TPS_filename); 
end
%--Calculate Descriptors --%
calculateDescriptorsInParallel(rounds);

%--Calculate Correspondences--%

parfor i =1:length(rounds)
    calcCorrespondences(rounds(i));
end

%--Register With Correspondences--%
parfor i = 1:length(rounds)
    registerWithCorrespondences(rounds(i),true); %   downsampled
    registerWithCorrespondences(rounds(i),false);%nondownsampled
end
