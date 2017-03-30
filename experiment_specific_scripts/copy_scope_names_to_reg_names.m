
INPUT_DIRECTORY = '/om/project/boyden/ExSeqAutoSlice/raw';
OUTPUT_DIRECTORY = '/om/project/boyden/ExSeqAutoSlice/input';

files = dir(fullfile(INPUT_DIRECTORY,'*.tif'));

EXPERIMENT_NAME  = 'exseqauto';

% sample pattern
% N-4_P_ will be round 0
% N-3_P_ will be round 1
% N-2_P will be round 2
% N-1_P will be round 3
% N_P will be round 4
% ...

for file_indx = 1:length(files)

    %Sample filename for this script: 'N-4_P_ch03.tif'
    parts = split(files(file_indx).name,'_');
    
    string_primer = parts{1};
    string_ligation= parts{2};
    string_color = parts{3};
    

    primer_string_pieces = split(string_primer,'N');
    offset_primer = 4;
    %'N' case is primer_string_pieces{2} ==0 
    if length(primer_string_pieces{2})>0
        offset_primer = offset_primer + str2num(primer_string_pieces{2});
    end
    
    %Count how many ligation rounds there have been
    multiplier_ligation = length(strfind(string_ligation,'L'));
    
    %Get the channel number from 'chXX.tif'
    string_color_parts = split(string_color,'.');
    color_chan_name = string_color_parts{1}; 
    
    %round number (add one to start at 1)
    round_number = multiplier_ligation*5 + offset_primer+1;
    %Produce the final string
    output_filename = sprintf('%s_round%i_%s.tif',EXPERIMENT_NAME,round_number,color_chan_name);
    
    src_file = fullfile(INPUT_DIRECTORY,files(file_indx).name);
    dest_file = fullfile(OUTPUT_DIRECTORY, output_filename);
    
    fprintf('Copying %s to %s\n',src_file,dest_file);
     %[status,message,messageId] = copyfile(src_file, dest_file)
    
end


