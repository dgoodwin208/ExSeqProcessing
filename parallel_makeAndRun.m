%Split up a directory of registered images into _1, _2, etc. 

params.registeredImagesDir;

%how many subsections to calculate the descriptors?
params.ROWS_DESC = 3;
params.COLS_DESC = 3;


filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_registered.tif',...
        params.FILE_BASENAME,1,params.CHAN_STRS{1}));
img = load3DTif_uint16(filename);

%chop the image up into grid
tile_upperleft_y = floor(linspace(1,size(img,1),params.ROWS_DESC+1));
tile_upperleft_x = floor(linspace(1,size(img,2),params.COLS_DESC+1));

 for x_idx=1:params.COLS_DESC
        for y_idx=1:params.ROWS_DESC
            tile_counter = tile_counter+1;
            
            outputdir = fullfile(params.registeredImagesDir,sprintf('subpiece%i',tile_counter));
            
            disp(['Running on row ' num2str(y_idx) ' and col ' num2str(x_idx) ]);
            
            %Make sure the folders for the descriptor outputs exist:
            descriptor_output_dir = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_%s/',params.SAMPLE_NAME,run_num,register_channel{1}));
            if exist(descriptor_output_dir,'dir')==0
                mkdir(descriptor_output_dir);
            end
            
            % get region, indexing column-wise
            ymin = tile_upperleft_y(y_idx);
            ymax = tile_upperleft_y(y_idx+1);
            xmin = tile_upperleft_x(x_idx);
            xmax = tile_upperleft_x(x_idx+1);
            
            %Calculate the features on the larger (overlapping) regions
            %         if length(img_cache) > 0
            %             tile_img = img_cache(tile_counter);
            %         else
            % create overlap region for calcuating features
            % will remove all points in the overlap region after calculation
            % but this avoids edge effects on any boundaries of
            
            
            ymin_overlap = floor(max(tile_upperleft_y(y_idx)-(params.OVERLAP/2)*(ymax-ymin),1));
            ymax_overlap = floor(min(tile_upperleft_y(y_idx+1)+(params.OVERLAP/2)*(ymax-ymin),size(img,1)));
            xmin_overlap = floor(max(tile_upperleft_x(x_idx)-(params.OVERLAP/2)*(xmax-xmin),1));
            xmax_overlap = floor(min(tile_upperleft_x(x_idx+1)+(params.OVERLAP/2)*(xmax-xmin),size(img,2)));
            
            for rnd_idx = 1:params.NUM_ROUNDS
                for c_idx = 1:params.NUM_CHANNELS
                    
                    filename = fullfile(params.registeredImagesDir,sprintf('%s_round%03d_%s_registered.tif',...
                        params.FILE_BASENAME,rnd_idx,params.CHAN_STRS{c_idx}));
                    img = load3DTif_uint16(filename);
            
                    img_crop = img(ymin_overlap:ymax_overlap,xmin_overlap:xmax_overlap,:);
                    
                    outputfile_name = fullfile(outputdir,sprintf('%s_round%03d_%s_registered.tif',...
                        params.FILE_BASENAME,rnd_idx,params.CHAN_STRS{c_idx}));
                    save3DTif_uint16(img_crop,outputfile_name);
                    
                end
            end
            
        end
 end