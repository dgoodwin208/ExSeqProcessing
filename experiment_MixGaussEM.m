
%Load the 3D Tif image
img = load3DTif_uint16('/Users/Goody/Neuro/ExSeq/exseq20170524/4_registration-cropped/exseqautoframe7crop_round009_ch00_registered.tif');
img = img(101:200,101:200,:);
img_size = size(img);

%load the punctafeinder results so we know 
pf_results = load3DTif_uint16('/Users/Goody/Neuro/ExSeq/exseq20170524/4_registration-cropped/exseqautoframe7crop_round009_ch00_L.tif');
pf_crop = pf_results(101:200,101:200,:);
num_found_puncta = length(unique(pf_crop(:)))-1;
%% Clamp the top 5% of the data
img = img(:); %vectorize
clamp_val = quantile(img,.995);
img(img>=clamp_val)=clamp_val;

%Divide each pixel by the max intensity value
pimg = img/clamp_val;

%% Draw points using a probability guess and some iterations
%Map onto a 0-centered normal distribution, and calculate the cdf of the
%negative value, then multiply by 2 bc we're only sampling one side

%since we're squashing everything to 0-1, sigma 
sigma = 1/(sqrt(2*pi)*1.);
sig_sqrt = sqrt(1/(2*sigma));

pimg_converted = -1*sig_sqrt*sqrt(log(1./pimg));
p = 1*normcdf(pimg_converted);
%%
iter=5;
num_pixels = length(pimg);
X_total = [];
for rnd = 1:iter
    uniform_random_numbers = rand(num_pixels,1);
    iteration_indices = 1:num_pixels;
    points_drawn = iteration_indices(uniform_random_numbers<p & p>.2);

    fprintf('%i points drawn\n',length(points_drawn));
    
    [i1 i2 i3] = ind2sub(img_size,points_drawn);
    X = [i1' i2' i3'];    
    
    %add a tiny amount of noise just to avoid repeats of the same pixel
    X_noise = X + normrnd(0,.3,size(X,1),size(X,2));
    
    X_total = [X_total; X_noise];

end

% % Create an Nx3 vector of all the points
% [i1 i2 i3] = ind2sub(img_size,indices);
% X = [i1' i2' i3'];

figure; scatter3(X_total(:,1),X_total(:,2),X_total(:,3))
%con

%% Run mixGaussEM using the "cheating" knowledge of how many puncta from the punctafeinder
idx = kmeans(X_total', num_found_puncta);

[label, model, llh] = mixGaussEm(X_total', idx);

obj = gmdistribution(model.mu',model.Sigma,model.w);
fprintf('Done. \n');
%% Plot

figure; 
% subplot(2,1,1);
scatter3(X_total(:,1),X_total(:,2),X_total(:,3),'.')
% subplot(2,1,2);
% hold on;
% scatter3(model.mu(1,:),model.mu(2,:),model.mu(3,:),'r')

hold on;
for i = 1:num_found_puncta
plot_gaussian_ellipsoid(model.mu(:,i), model.Sigma(:,:,i))
% pause
end

%% 

simulated_data = makeSimulatedRound(num_found_puncta,...
                ones(num_found_puncta,1),...
                eye(4),...
                round(model.mu'),...
                model.Sigma,...
                0,... %puncta variability
                img_size);
