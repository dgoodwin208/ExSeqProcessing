function shift = phaseOnlyCorrelation(baseImg,moveImg,offsets)

    %border_mask = zeros(size(baseImg));
    %border_mask(1:offsets(1)+1,:,:)=1; border_mask(end-offsets(1):end,:,:)=1;
    %border_mask(:,1:offsets(2)+1,:)=1; border_mask(:,end-offsets(2):end,:)=1;
    %border_mask(:,:,1:offsets(3)+1)=1; border_mask(:,:,end-offsets(3):end)=1;
    %border_mask = logical(border_mask);
    %baseImg(border_mask) = 0.;
    %moveImg(border_mask) = 0.;

    % hann window
    size_wx = size(baseImg,1);
    size_wy = size(baseImg,2);
    size_wz = size(baseImg,3);
    [wx,wy,wz] = ndgrid(1:size_wx,1:size_wy,1:size_wz);
    w = sqrt((wx./size_wx-0.5).^2+(wy./size_wy-0.5).^2+(wz./size_wz-0.5).^2);
    clear wx wy wz

    hann_window = 0.5-0.5*cos(2*pi*(w+0.5));
    range = w <= 0.5;
    hann_window = hann_window.*range;
    clear w range

    f1 = fftn(baseImg.*hann_window);
    f2 = fftn(moveImg.*hann_window);
    r  = f1.*conj(f2);
    clear hann_window f1 f2

    r  = r./abs(r);
    ir = ifftn(r);
    clear r

    ir(offsets(1)+2:end-offsets(1)-1,:,:) = nan(1);
    ir(:,offsets(2)+2:end-offsets(2)-1,:) = nan(1);
    ir(:,:,offsets(3)+2:end-offsets(3)-1) = nan(1);
    %disp(ir(1:4,1:4,1:4))
    %disp(ir(end-3:end,end-3:end,end-3:end))

    [x,y,z] = ind2sub(size(baseImg),find(abs(ir)==max(abs(ir(:)))));
    clear ir
    disp([x,y,z]);

    shift = [x y z]-1;
    shift = shift-(shift>size(baseImg)*0.5).*size(baseImg);
    %disp(shift);

end

