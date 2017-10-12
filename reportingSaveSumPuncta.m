loadParameters;
chan_strs = {'ch00','ch01SHIFT','ch02SHIFT','ch03SHIFT'};

for round_num = 1:params.NUM_ROUNDS
    for chan_num = 1:params.NUM_CHANNELS
        tic
        chan_str = chan_strs{chan_num};
        filename_in = fullfile(params.punctaSubvolumeDir,sprintf('%s_round%.03i_%s_puncta.tif',params.FILE_BASENAME,round_num,chan_str));
        stack_in = load3DTif_uint16(filename_in);
        rounds(:,:,:,chan_num)=stack_in;
    end
    sum_rounds = sum(rounds,4);
    
    filename_out = fullfile(params.punctaSubvolumeDir,sprintf('%s_round%.03i_summedpuncta.tif',params.FILE_BASENAME,round_num,chan_str));
    save3DTif_uint16(sum_rounds,filename_out);
end

  