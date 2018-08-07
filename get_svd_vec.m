function [vec] = get_svd_vec(H)

[s,v,d] = svd(H);
vec = abs(s(:, 1));
end
