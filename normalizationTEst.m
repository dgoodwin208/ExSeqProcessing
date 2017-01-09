chan1 = load3DTif('../cropCulture/CROPTPSsa0916dncv_round1_chan1.tif');
chan2 = load3DTif('../cropCulture/CROPTPSsa0916dncv_round1_chan2.tif');
chan3 = load3DTif('../cropCulture/CROPTPSsa0916dncv_round1_chan3.tif');
chan4 = load3DTif('../cropCulture/CROPTPSsa0916dncv_round1_chan4.tif');

NUM_CHANNELS = 4;

data_cols(:,1) = reshape(chan1,[],1);    
data_cols(:,2) = reshape(chan2,[],1);    
data_cols(:,3) = reshape(chan3,[],1);    
data_cols(:,4) = reshape(chan4,[],1);    
%% 
%     %Normalize the data
data_cols_norm = quantilenorm(data_cols);

%%
chan1_norm = reshape(data_cols_norm(:,1),size(chan1));
chan2_norm = reshape(data_cols_norm(:,2),size(chan2));
chan3_norm = reshape(data_cols_norm(:,3),size(chan3));
chan4_norm = reshape(data_cols_norm(:,4),size(chan4));

%%
figure;
subplot(1,2,1);
imagesc(max(chan1,[],3));
subplot(1,2,2);
imagesc(max(chan1_norm,[],3));

%%
summed = chan1+chan2+chan3+chan4;
summed_norm = chan1_norm+chan2_norm+chan3_norm+chan4_norm;

figure;
subplot(1,2,1);
imagesc(max(summed,[],3)); axis off; colormap gray;
subplot(1,2,2);
imagesc(max(summed_norm,[],3)); axis off; colormap gray;