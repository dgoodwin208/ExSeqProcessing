% Code taken from Paul Scovanner's homepage: 
% http://www.cs.ucf.edu/~pscovann/


% Calculate 3D Sift
% img is the input in question.
% keypts is an Nx3 matrix of keypoints

function  keys = calculate_3DSIFT(img, keypts,skipDescriptor)

%By default, we calculate the descriptor
if nargin<3
    skipDescriptor=false;
end
%However, in the case when we only want the keypoint (ie for Shape Context)
%we skip the calclation of the SIFT descriptor to save time

keys = cell(size(keypts,1),1);
i = 0;
offset = 0;
while 1

    reRun = 1;
    i = i+1;
    
    while reRun == 1
        
        loc = keypts(i+offset,:);
        %fprintf(1,'Calculating keypoint at location (%d, %d, %d)\n',loc);
        
        % Create a 3DSIFT descriptor at the given location
        if ~skipDescriptor
            [keys{i} reRun] = Create_Descriptor(img,1,1,loc(1),loc(2),loc(3));
        else         
            clear k; reRun=0;
            k.x = loc(1); k.y = loc(2); k.z = loc(3);
            keys{i} = k;
        end

        if reRun == 1
            offset = offset + 1;
        end
        
        %are we out of data?
        if i+offset>=size(keypts,1)
            break;
        end
    end
    
    %are we out of data?
    if i+offset>=size(keypts,1)
            break;
    end
end
%remove any pre-initialized descriptors that weren't used
keys(i:end)=[];

fprintf(1,'\nFinished.\n%d points thrown out do to poor descriptive ability.\n',offset);

end
