spotdet_regionwise does: 
DoG then threshold based on max DoG value of bkgd image then watershed then normalize pxls in foreground and look at each intersection of watershed regions and whoever has the highest mean value wins so (--[**)++] if ++ is greater than -- and ** is the intersection then the whole thing becomes (  [++)++]. 

spotdet_pxlwise isnt that great but does: 
DoG then threshold based on max DoG val of bkgd image then compares normalized pxls on the raw images btwn channels and then watersheds 