clear all;
load('splintr3F2_results.mat');
output_dir = '/Users/goody/Neuro/ExSeq/splintr/3F2Morphology/';
IMG_SIZE  = [2048 2048 100];

%% First create a 3D image of all the puncta in space 
output_img = zeros(IMG_SIZE);

for p_idx = 1:length(transcript_objects)
    output_img(transcript_objects{p_idx}.voxels) = 500;
end
    
save3DTif_uint16(output_img, fullfile(output_dir,'segmentedPuncta.tif'));


%% Extract the gene names and centroids from the transcript_objects

genename = cell(length(transcript_objects),1);
puncta_centroids = zeros(length(transcript_objects),3);

for idx = 1:length(transcript_objects)
    genename{idx} = transcript_objects{idx}.name;
    puncta_centroids(idx,:) = transcript_objects{idx}.pos;
end

%% Write the XML

docNode = com.mathworks.xml.XMLUtils.createDocument... 
    ('root_element')
docNode.appendChild(docNode.createComment('First draft of XML for ExSeq, written by Dan Goodwin dgoodwin@mit.edu'));
docRootNode = docNode.getDocumentElement;
docRootNode.setAttribute('ExperimentName','ExSeq');
docRootNode.setAttribute('ExperimentDate','2019-06-10');
docRootNode.setAttribute('SpatialUnits','Pixels');
for i=1:length(transcript_objects)
    thisElement = docNode.createElement('insitu_read'); 
    thisElement.setAttribute('x',sprintf('%i',round(puncta_centroids(i,2))));
    thisElement.setAttribute('y',sprintf('%i',round(puncta_centroids(i,1))));
    thisElement.setAttribute('z',sprintf('%i',round(puncta_centroids(i,3))));
    thisElement.appendChild... 
        (docNode.createTextNode(sprintf('%s',genename{i})));
    docRootNode.appendChild(thisElement);
end


xmlFileName = 'exseq_genelocations_v1.xml';
xmlwrite(fullfile(output_dir,xmlFileName),docNode);
type(xmlFileName);


