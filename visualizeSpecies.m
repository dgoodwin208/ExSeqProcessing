 CStr = textread('filtered_transcripts.txt', '%s', 'delimiter', '\n');
 
 
 data = load3DTif(fullfile(dir_input,files(1).name));
 
 gene_idx = 1;
 gene_names = {};
 
 for row_idx = 1:length(CStr)
    entry = CStr{row_idx};
    elts = split(entry);
    
    %If it's a new species
    if strcmp(elts{1},'Colors:')
    
        gene_name = elts{6};
        
    end
 end