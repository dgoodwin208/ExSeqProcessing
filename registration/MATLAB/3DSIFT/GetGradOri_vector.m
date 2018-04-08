function [mag vect precomp_grads, yy, ix] = GetGradOri_vector(pix,r,c,s, fv, sift_params, precomp_grads)

rows = sift_params.pix_size(1);
cols = sift_params.pix_size(2);
slices = sift_params.pix_size(3);

% clamp r,c,s to the bounds of the img (pix)
if r < 1
    r = 1;
end
if c < 1
    c = 1;
end
if s < 1
    s = 1;
end
if r > rows
    r = rows;
end
if c > cols
    c = cols;
end
if s > slices
    s = slices;
end

% Check if computed the gradient previously before
key = sub2ind(sift_params.pix_size, r,c,s);
if isKey(precomp_grads, key)
    val = precomp_grads(key); 
    val.count = val.count + 1; %increment counter
    % retrieve the data
    mag = val.mag;
    vect = val.vect;
    yy = val.yy;
    ix = val.ix;
    precomp_grads(key) = val; % save updated before returning
    return
end

if (c == 1)
    xgrad = 2.0 * (double(pix(r,c+1,s)) - double(pix(r,c,s)));
elseif (c == cols)
    xgrad = 2.0 * (double(pix(r,c,s)) - double(pix(r,c-1,s)));
else
    xgrad = double(pix(r,c+1,s)) - double(pix(r,c-1,s));
end
if (r == 1)
    ygrad = 2.0 * (double(pix(r,c,s)) - double(pix(r+1,c,s)));
elseif (r == rows)
    ygrad = 2.0 * (double(pix(r-1,c,s)) - double(pix(r,c,s)));
else
    ygrad = double(pix(r-1,c,s)) - double(pix(r+1,c,s));
end
if (s == 1)
    zgrad = 2.0 * (double(pix(r,c,s+1)) - double(pix(r,c,s)));
elseif (s == slices)
    zgrad = 2.0 * (double(pix(r,c,s)) - double(pix(r,c,s-1)));
else
    zgrad = double(pix(r,c,s+1)) - double(pix(r,c,s-1));
end

xgrad = double(xgrad);
ygrad = double(ygrad);
zgrad = double(zgrad);

mag = sqrt(xgrad * xgrad + ygrad * ygrad + zgrad * zgrad);

if mag ~=0
    vect = [xgrad ygrad zgrad] ./ mag;
else
    vect = [1 0 0];
end

%Find the nearest tesselation face indices
corr_array = fv.centers * vect';
[yy ix] = sort(corr_array,'descend');

val = {}; % number of times seen 1
val.count = 1; % number of times seen 1
val.mag = mag;
val.vect = vect;
val.yy = yy(1:sift_params.Tessel_thresh, :);
val.ix = ix(1:sift_params.Tessel_thresh);
precomp_grads(key) = val;
return
