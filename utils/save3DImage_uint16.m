function save3DImage_uint16(vol,path)

if max(vol(:))>intmax('uint16')
     fprintf('Warning: some values to be saved at %s are over uint16 maxval\n',path)
end
if min(vol(:))<0
     fprintf('Warning: some values to be saved at %s less than 0\n',path)
end

if endsWith(path,'.tif')

    %This is a wrapper function around the saveastiff fo
    opts_tiff.big = false;
    opts_tiff.append=false;
    opts_tiff.overwrite = true;

    saveastiff(uint16(vol),path,opts_tiff);

elseif endsWith(path,'.h5')

    try
        h5create(path,'/image',size(vol),'DataType','uint16');
    catch ME
        fprintf('Warning: %s\n%s.\n',path,ME.message);
    end
    h5write(path,'/image',uint16(vol));

end

end
