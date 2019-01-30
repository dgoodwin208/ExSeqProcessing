function writeCSVfromTranscriptObjects(transcript_objects,filename)
%Write all the aligned transcripts to a csv
DELIMITER = char(9);

fileID = fopen(filename,'w');
for t_idx= 1:length(transcript_objects)
    transcript = transcript_objects{t_idx};
    if isfield(transcript,'name')
        fprintf(fileID,'%i \t %s \t',t_idx,mat2str(transcript.img_transcript));
        
        nameparts = split(transcript.name,DELIMITER);
        
        %If it's a unique match, use the name
        for part_idx = 1:length(nameparts)
            fprintf(fileID,sprintf('%s \t',nameparts{part_idx}));
        end
        row_string = fprintf(fileID,'\n');

    end
end

% fprintf(fileID,output_csv);
fclose(fileID);


end

