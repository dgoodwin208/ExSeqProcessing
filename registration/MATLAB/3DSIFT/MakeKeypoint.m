function [key, precomp_grads] = MakeKeypoint(pix, xyScale, tScale, x, y, z, sift_params, precomp_grads)
    k.x = x;
    k.y = y;
    k.z = z;
    k.xyScale = xyScale;
    k.tScale = tScale;
    [key precomp_grads] = MakeKeypointSample(k, pix, sift_params, precomp_grads);
    return;
end


function [key precomp_grads] = MakeKeypointSample(key, pix, sift_params, precomp_grads)


MaxIndexVal = 0.2;
changed = 0;

[vec precomp_grads] = KeySampleVec(key, pix, sift_params, precomp_grads);
VecLength = length(vec);

%fprintf('ML\n');
%for i=1:VecLength
    %if (vec(i) ~= 0 )
        %fprintf('index[%d]=%.4f, ', i - 1, vec(i));
    %end
%end
%fprintf('\n');

vec = NormalizeVec(vec, VecLength);

for i = 1:VecLength
    if (vec(i) > MaxIndexVal)
        vec(i) = MaxIndexVal;
        changed = 1;
    end
end
if (changed)
    vec = NormalizeVec(vec, VecLength);
end

for i = 1:VecLength
    intval = int16(512.0 * vec(i));
    if ~(intval >= 0)
        disp('Assertation failed in MakeKeypoint.m');
    end
    key.ivec(i) = uint8(min(255, intval));
end
end


