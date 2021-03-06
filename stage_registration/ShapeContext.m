function F=ShapeContext(Pointsc,Points)
%Calaculate ShapeContext vectors around Pointsc
%Points is what is calculated in the histogram for Pointsc
defaultoptions=struct('r_max',3,'r_min',1e-5,'r_bins',10,'a_bins',10,'rotate',0,'method',1,'maxdist',5);
options=defaultoptions;
F=getHistogramFeatures(Pointsc,Points,options);
return 

%For each point in keypoint, calculate the shapecontext descriptor
%by counting the points
%In our application, 'keypoints' will be the keypoints used for the SIFT d
function F=getHistogramFeatures(keypoints,points,options)

options.w_bins=options.a_bins;
F=zeros(options.w_bins*options.a_bins*options.r_bins,size(keypoints,1));

for i=1:size(keypoints,1)
    % The Current Point
    P=keypoints(i,:);
    
    % Determine the log-distance an angle of all points
    % relative to the current point
    O=bsxfun(@minus,points,P);
    R=sqrt(sum(O.^2,2));
    R(R<options.r_min)=options.r_min;  % not effective?
    
    RMAX = 100; %options.r_max;
    RMIN = 3; %options.r_min;
    keep_indices = R<RMAX & R>RMIN;
    R = R(keep_indices);
    R = (R-RMIN)/(RMAX-RMIN);

    O = O(keep_indices,:);
    
%     A = (atan2(O(:,1),O(:,2))+pi)/(2*pi);
%     r = sqrt(sum(O(:,1:2).^2,2));
%     W = (atan2(O(:,3),r)+pi/2)/pi;
    A = (atan2(O(:,1),O(:,3))+pi)/(2*pi);
    r = sqrt(sum(O(:,[1 3]).^2,2));
    W = (atan2(O(:,2),r)+pi/2)/pi;
    
    
    % Histogram positions
    Abin=A*(options.a_bins-1e-10)+1;
    Wbin=W*(options.w_bins-1e-10)+1;
    Rbin=R*(options.r_bins-1e-10)+1;
    
    % Construct the polar-distance histogram by counting
    Abin=floor(Abin); 
    Wbin=floor(Wbin); 
    Rbin=floor(Rbin);

    H = zeros([options.r_bins,options.a_bins,options.w_bins]);
    for j = 1:length(Wbin)
        H(Rbin(j),Abin(j),Wbin(j)) = H(Rbin(j),Abin(j),Wbin(j))+1;
    end
    
    F(:,i)=H(:)./sum(keep_indices);
    
end



