for roundnum = 1:12
        chan1 = load3DTif(sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan1.tif',roundnum));
        chan2 = load3DTif(sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan2.tif',roundnum));
        chan3 = load3DTif(sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan3.tif',roundnum));
        chan4 = load3DTif(sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan4.tif',roundnum));

        data_cols(:,1) = reshape(chan1,[],1);
        data_cols(:,2) = reshape(chan2,[],1);
        data_cols(:,3) = reshape(chan3,[],1);
        data_cols(:,4) = reshape(chan4,[],1);

        %     %Normalize the data
        data_cols_norm = quantilenorm(data_cols);

        % reshape the normed results back into 3d images
        chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
        chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
        chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
        chan4_norm = reshape(data_cols_norm(:,4),size(chan4));


        summed = chan1+chan2+chan3+chan4;
        summed_norm = chan1_norm+chan2_norm+chan3_norm+chan4_norm;

        save3DTif(summed_norm,sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_summedNorm.tif',roundnum));

        save3DTif(chan1_norm, sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan1Norm.tif',roundnum));
        save3DTif(chan2_norm, sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan2Norm.tif',roundnum));
        save3DTif(chan3_norm, sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan3Norm.tif',roundnum));
        save3DTif(chan4_norm, sprintf('/om/project/boyden/ExSeqSlice/input/sa0916slicedncv_round%i_chan4Norm.tif',roundnum));

end