clear all;

%% Manually specify ROI names in 0_raw folder

ROInames = {

%     'S1ROI1';
%     'S1ROI2';
%     'S1ROI3';
%     'S1ROI4';
% %     'S1ROI5';
%     
%     'S2ROI1';
%     'S2ROI2';
%     'S2ROI3';
%     'S2ROI4';   
% 
%     'S3ROI1';
%     'S3ROI2';
%     'S3ROI3';
%     'S3ROI4';  
% %     
%     'S4ROI1';
%     'S4ROI2';
%     'S4ROI3';
%     'S4ROI4';       
%     'S1ROI1';
%     'S1ROI2';
%     'S1ROI3';
%     'S1ROI4';
    
%     'S2ROI1';
%     'S2ROI2';
%     'S2ROI3';
%     'S2ROI4';

%     '5xFAD-ctx-ROI1';
%     '5xFAD-ctx-ROI2';
%     '5xFAD-ctx-ROI3';
%     '5xFAD-ctx-ROI4';
%     '5xFAD-ctx-ROI5';
%     
%     'WT-ctx-ROI1';
%     'WT-ctx-ROI2';
%     'WT-ctx-ROI3';
%     'WT-ctx-ROI4';
%     'WT-ctx-ROI5';    
    % 'ROI1';
%     'ROI2';
%     'ROI3';
%     'ROI4';
%     'ROI5';
% %     'ROI6';
% %     'ROI7';
%     'ROI8'


    };

%% Loop through and run pipeline for each ROI
tic
for rr = 1:length(ROInames)
    disp(ROInames{rr})
    
    %load parameters for each ROI
    editParams_batch(ROInames{rr});
    
    % Step one: downsampling the data (useful for registration) and correct for
    % chromatic shifts between the color channels (if any)
    %stage_downsampling_and_color_correction;
    stage_downsampling_and_color_correction;

    % Step two: combine the color channels, which is useful for registration
    % and puncta detection
    % stage_normalization;

    % Step three: register the data to a common coordinate space. You will pick
    % a round in params.REFERENCE_ROUND_WARP
    stage_registration;
end
toc
