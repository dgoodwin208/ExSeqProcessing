% makeBarcodeMultiPlots

function makeBarcodeMultiPlots(registrationDir,transcriptsDir,numRounds,channelsList,barcode,saveDir)

    loadParameters;

    base0 = load(fullfile(transcriptsDir,sprintf('%s_transcriptsv9.mat',params.FILE_BASENAME)));
    base1 = load(fullfile(transcriptsDir,sprintf('%s_transcripts_probfiltered.mat',params.FILE_BASENAME)));

    dist = pdist2(base1.transcripts_probfiltered,barcode);
    dist_min = min(dist,[],2);
    barcode_pos_valid = base0.pos(base1.puncta_indices_probfiltered(dist_min==0),:);
    barcode_pos_invalid = base0.pos(base1.puncta_indices_probfiltered(dist_min>0),:);

    f = figure;
    f.Position=[1000 100 1200 1000];

    count = 1;
    for r_i = 1:numRounds
        for c_i = 1:length(channelsList)
            disp(sprintf('round%03i_%s',r_i,channelsList{c_i}));

            disp('loading tif');
            tic;
            img = load3DTif(fullfile(registrationDir,sprintf('%s_round%03i_%s_registered.tif',...
                params.FILE_BASENAME,r_i,channelsList{c_i})));
            toc;

            img = double(img);
            img = img ./ max(img(:));
            [x,y,z] = ndgrid(1:size(img,1),1:size(img,2),1:size(img,3));

            img_barcode = zeros(size(img));
            for b_i = 1:size(barcode_pos_valid,1)
                barcode_pos = barcode_pos_valid(b_i,:);
                img_barcode(barcode_pos(1),barcode_pos(2),barcode_pos(3)) = 1;
            end
            img_barcode = imgaussfilt3(img_barcode);
            fore = img_barcode>0;

            subplot(numRounds,length(channelsList),count);
            if count == 1
                title(sprintf('%s, r%03i,%s',num2str(barcode,'%d'),r_i,channelsList{c_i}));
            else
                title(sprintf('r%03i,%s',r_i,channelsList{c_i}));
            end
            xlabel('X'); ylabel('Y'); zlabel('Z');
            whitebg('black');
            hold on; box on;

            disp('plotting puncta');
            tic;
            axis([0 size(img,1)*1.1 0 size(img,2)*1.1 0 size(img,3)*1.1]);
            scatter3(x(fore), y(fore), z(fore), 1, img(fore) * [1, 1, 1] .* 10);
            toc;

            count = count+1;
        end
    end

    if exist(saveDir) == 0
        mkdir(saveDir)
    end

    f.InvertHardcopy = 'off';
    saveas(f,fullfile(saveDir,sprintf('%s_barcode_%s_multiplots.png',params.FILE_BASENAME,num2str(barcode,'%d'))));
end
