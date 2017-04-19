% make ROIs instread of SegmentGUI

function makeROIs(varargin)

    if ~isempty(varargin)
        fileToStart = varargin{1};
    else
        fileToStart = 1;
    end

    Hs = struct;

    % Start by checking if we already have 'data***.mat' files
    % that store |image_object|. If so, we load the data files
    % to get images and the segmentation ROIs. 
    %---------------------------------------------------------
    Hs.dirPath = pwd;
    [Hs.dataFiles,Hs.dataNums] = getDataFiles(Hs.dirPath);

    % Look for image files in the current working directory. If we
    % don't find any, ask the user to navigate to the image files
    % using the GUI file browser. We check again for 'data***.mat'
    % and then the image files 
    %--------------------------------------------------------------
    [Hs.foundChannels,Hs.fileNums,Hs.imgExts] = getImageFiles(Hs.dirPath);
    if isempty(Hs.fileNums)  % no image files here
        fprintf(1,'Could not find any image files!\n');
        return;  % user did not want to navigate to image files, quit
    end

    % make sure that if we have both data & image files, that the numbering
    % scheme makes some sense. For example, if we have 'data001-data010.mat'
    % files then there should also be 'tmr001-tmr010.tif' and not less.
    %------------------------------------------------------------------------
    if numel(Hs.dataFiles) > numel(Hs.fileNums)
        msg = sprintf(['There are more data*.mat files than image files\n'...
                       'This should not happen']);
        error(msg);
    end

    % remove dapi & trans from Hs.foundChannels and imgExts to get the 
    % RNA channels only
    Hs.RNAchannels = Hs.foundChannels;
    Hs.RNAchannels(strcmp('dapi',Hs.RNAchannels)) = [];
    Hs.RNAchannels(strcmp('trans',Hs.RNAchannels)) = [];
    Hs.RNAchannels = [Hs.RNAchannels 'NONE'];  % this selection hides RNA
    Hs.RNAchannel = Hs.RNAchannels{1};  % use the first RNA channel str to start

    %start at first or user-specified RNA file
    if fileToStart > length(Hs.fileNums)
        Hs.fileNum = Hs.fileNums(end);
    else
        Hs.fileNum = Hs.fileNums(fileToStart); 
    end

    Hs = loadFileSet(Hs);

    % make ROIs fro all images
    for fileNum = Hs.fileNums(fileToStart:length(Hs.fileNums))
        Hs.fileNum = fileNum;

        disp(['[',num2str(fileNum),'] segmentation..'])
        Hs = segmentObject_Callback(Hs);

        disp(['[',num2str(fileNum),'] save data..'])
        Hs = nextFileB_Callback(Hs);
    end

    return;


function Hs = loadFileSet(Hs)
    % Check for data files, get the objects & segmentation boundaries
    %-----------------------------------------------------------------
    Hs.allMasks = [];
    dataInd = find(Hs.fileNum == Hs.dataNums);  % matches data00N w/ tmr00N only
    if isempty(dataInd) 
        Hs.currObjs = [];
    else 
        Hs.currObjs = load([Hs.dirPath filesep Hs.dataFiles(dataInd).name]);
        Hs.currObjs = Hs.currObjs.objects;
        for obj = Hs.currObjs
%             assert(isa(obj, 'improc2.dataNodes.GraphBasedImageObject'), ...
%                 'Convert the already-segmented objects in this directory to Graph Based Image Objects first.');
            if isempty(Hs.allMasks)  % first object
                Hs.allMasks = obj.object_mask.imfilemask;
            else
                Hs.allMasks = cat(3,Hs.allMasks,obj.object_mask.imfilemask);
            end
        end
    end

    % Get max-merges of the stacks and display the image 
    Hs = getMaxes(Hs,true);
    Hs.imgH = [];


function Hs = getMaxes(Hs,updateAllImgs)
% RNA and DAPI images max merges are a sampling of the image stack planes
% trans is the 3rd plane in the stack
    
    currInd = strcmp(Hs.RNAchannel,Hs.foundChannels);
    if ~strcmp(Hs.RNAchannel,'NONE')
        Hs.RI = readmm(sprintf('%s%s%s%03d%s',...
                   Hs.dirPath,filesep,Hs.RNAchannel,Hs.fileNum,Hs.imgExts{currInd}));
        Hs.RI = Hs.RI.imagedata; 
        Hs.RI = scale(max(Hs.RI(:,:,round(linspace(3,size(Hs.RI,3),10))),[],3));
        Hs.RI2 = scale(medfilt2(Hs.RI));
    end

    if updateAllImgs
        sz = size(Hs.RI);
        ty = class(Hs.RI);
        tF = find(strcmpi('trans',Hs.foundChannels));
        if isempty(tF)
            Hs.TI = zeros(sz,ty);
%            set(Hs.transCheck,'Enable','off','Value',0);
        else
            Hs.TI = readmm(sprintf('%s%s%s%03d%s',Hs.dirPath,filesep,...
                            Hs.foundChannels{tF},Hs.fileNum,Hs.imgExts{tF}),3);
            Hs.TI = scale(Hs.TI.imagedata);
%            set(Hs.transCheck,'Enable','on');
        end

        dF = find(strcmpi('dapi',Hs.foundChannels));
        if isempty(dF)
            Hs.DI = zeros(sz,ty);
%            set(Hs.dapiCheck,'Enable','off','Value',0);
        else
            Hs.DI = readmm(sprintf('%s%s%s%03d%s',Hs.dirPath,filesep,...
                            Hs.foundChannels{dF},Hs.fileNum,Hs.imgExts{dF}));
            Hs.DI = Hs.DI.imagedata;
            Hs.DI = scale(max(Hs.DI(:,:,round(linspace(3,size(Hs.DI,3),4))),[],3));
%            set(Hs.dapiCheck,'Enable','on');
        end
    end


function retHs = segmentObject_Callback(Hs)

%    hROI = imfreehand;  % add a new freehand 

    % Use the ROI mask binary image to append a new |image_object|  and
    % to update the image axes 
%    maskImg = hROI.createMask;
    maskImg = ones(size(Hs.RI),'logical');
    cc = bwconncomp(maskImg);
    if sum(maskImg(:)) < 50  % user drew a tiny (NULL) ROI
        fprintf(1,'Segmented region was too small to create object\n');

        return
    end

    % should have one ROI, create & append an |image_object|
    fnumStr = sprintf('%03d',Hs.fileNum);
    
    newObj = improc2.buildImageObject(maskImg, fnumStr, Hs.dirPath);
    
    Hs.currObjs = [Hs.currObjs, newObj];
    
    Hs.allMasks = cat(3,Hs.allMasks,maskImg); % Store with other masks.

    retHs = Hs;


function retHs = nextFileB_Callback(Hs)

    % Save the |image_object|s to data***.mat file
    objects = Hs.currObjs;
    save(sprintf('%s%sdata%03d.mat',Hs.dirPath,filesep,Hs.fileNum),'objects');
    [Hs.dataFiles,Hs.dataNums] = getDataFiles(Hs.dirPath);
    clear objects;
    Hs.currObjs = [];

    nextInd = find(Hs.fileNum == Hs.fileNums) + 1;
    if nextInd > numel(Hs.fileNums) % No files left, close the GUI.
        retHs = Hs;
        return;
    end

    % load the next set of data
    Hs.fileNum = Hs.fileNums(nextInd);
    Hs = loadFileSet(Hs);

    retHs = Hs;

