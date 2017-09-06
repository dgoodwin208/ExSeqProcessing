function save3DTif_uint16(vol,path)

%This is a wrapper function around the saveastiff fo
opts_tiff.big = false;
opts_tiff.append=false;
opts_tiff.overwrite = true;

if max(vol(:))>intmax('uint16')
     fprintf('Warning: some values to be saved at %s are over uint16 maxval\n',path)
end
if min(vol(:))<0
     fprintf('Warning: some values to be saved at %s less than 0\n',path)
end

saveastiff(uint16(vol),path,opts_tiff);

end
