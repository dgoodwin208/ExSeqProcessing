% makeBarcode3DPlots

function makeBarcode3DPlots(transcriptsDir,groupList,saveDir)

    loadParameters;
    load('groundtruth_codes.mat');

    f = figure;
    f.Position=[1000 100 900 900];

    xlabel('X'); ylabel('Y'); zlabel('Z');
    whitebg('black');

    for i = 1:length(groupList)
        disp(['group=',groupList{i}])
        base0 = load(fullfile(transcriptsDir,groupList{i},sprintf('%s_transcriptsv9.mat',params.FILE_BASENAME)));
        base1 = load(fullfile(transcriptsDir,groupList{i},sprintf('%s_transcripts_probfiltered.mat',params.FILE_BASENAME)));

        dist = pdist2(base1.transcripts_probfiltered,groundtruth_codes);
        dist_min = min(dist,[],2);
        base_pos_valid = base0.pos(base1.puncta_indices_probfiltered(dist_min==0),:);
        base_pos_invalid = base0.pos(base1.puncta_indices_probfiltered(dist_min>0),:);

        img = load3DTif(fullfile(transcriptsDir,groupList{i},'alexa001.tiff'));
        img = double(img);
        img = img ./ max(img(:));
        fore = img > 0.1;
        [x,y,z] = ndgrid(1:size(img,1),1:size(img,2),1:size(img,3));
        scatter3(x(fore), y(fore), z(fore), 1, img(fore) * [1, 1, 1] .* 2);
        hold on; box on;
        scatter3(base_pos_valid(:,1),base_pos_valid(:,2),base_pos_valid(:,3),5,'cyan');
        scatter3(base_pos_invalid(:,1),base_pos_invalid(:,2),base_pos_invalid(:,3),3,'red');
        pbaspect(size(img) .* [1, 1, 4]);

        if exist(fullfile(saveDir,groupList{i})) == 0
            mkdir(fullfile(saveDir,groupList{i}));
        end
        view(2)
        f.InvertHardcopy = 'off';
        saveas(f,fullfile(saveDir,groupList{i},sprintf('%s_%s_barcode3dplots-xy.png',params.FILE_BASENAME,groupList{i})));
        view(0,0)
        saveas(f,fullfile(saveDir,groupList{i},sprintf('%s_%s_barcode3dplots-xz.png',params.FILE_BASENAME,groupList{i})));
        view(90,0)
        saveas(f,fullfile(saveDir,groupList{i},sprintf('%s_%s_barcode3dplots-yz.png',params.FILE_BASENAME,groupList{i})));
        view(30,20)
        saveas(f,fullfile(saveDir,groupList{i},sprintf('%s_%s_barcode3dplots-3d.png',params.FILE_BASENAME,groupList{i})));

        hold off
    end
end

