%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function calculateShapeContextDescriptors(fixed_run)

loadExperimentParams;

disp(['FIXED: ' num2str(fixed_run)])

all_sc_existed = true;
for x_idx=1:params.COLS_TFORM
    for y_idx=1:params.ROWS_TFORM
        df_sift_norm_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_df_sift_norm_r%uc%u.bin',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
        if ~exist(df_sift_norm_filename)
            disp([df_sift_norm_filename,' is not existed.']);
            all_sc_existed = false;
            break;
        end

        df_sc_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_df_sc_r%uc%u.bin',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
        if ~exist(df_sc_filename)
            disp([df_sc_filename,' is not existed.']);
            all_sc_existed = false;
            break;
        end

        df_sift_sc_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_df_sift_sc_r%uc%u.bin',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
        if ~exist(df_sift_sc_filename)
            disp([df_sift_sc_filename,' is not existed.']);
            all_sc_existed = false;
            break;
        end
    end
end

if all_sc_existed
    lf_sift_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_lf_sift_r%uc%u.mat',...
        params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
    if exist(lf_sift_filename)
        disp('all sift sc files are existed.');
        return;
    end
end

filename = fullfile(params.INPUTDIR,sprintf('%sround%03d_%s.tif',...
    params.SAMPLE_NAME,fixed_run,params.CHANNELS{1} ));

imgFixed_total = load3DTif_uint16(filename);
imgFixed_total_size = size(imgFixed_total);


%LOAD FILES WITH CROP INFORMATION, CROP LOADED FILES
cropfilename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_cropbounds.mat',params.SAMPLE_NAME,fixed_run));
if exist(cropfilename,'file')==2
    load(cropfilename,'bounds'); bounds_fixed = floor(bounds); clear bounds;
    imgFixed_total = imgFixed_total(bounds_fixed(1):bounds_fixed(2),bounds_fixed(3):bounds_fixed(4),:);
end

%------------------------------Load Descriptors -------------------------%
%Load all descriptors for the FIXED channel
tic;
keys_fixed_total_sift.pos = [];
keys_fixed_total_sift.ivec = [];
for register_channel = [params.REGISTERCHANNELS_SIFT]
    descriptor_output_dir_fixed = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_%s/',params.SAMPLE_NAME, ...
        fixed_run,register_channel{1}));

    files = dir(fullfile(descriptor_output_dir_fixed,'*.mat'));

    for file_idx= 1:length(files)
        filename = files(file_idx).name;

        %The data for each tile is keys, xmin, xmax, ymin, ymax
        data = load(fullfile(descriptor_output_dir_fixed,filename));
        keys = vertcat(data.keys{:});
        pos = [[keys(:).y]'+data.ymin-1,[keys(:).x]'+data.xmin-1,[keys(:).z]'];
        ivec = vertcat(keys(:).ivec);

        keys_fixed_total_sift.pos  = vertcat(keys_fixed_total_sift.pos,pos);
        keys_fixed_total_sift.ivec = vertcat(keys_fixed_total_sift.ivec,ivec);
    end
end
fprintf('load sift keys of fixed round%03d (mod). ',fixed_run);toc;

tic;
keys_fixed_total_sc.pos = [];
for register_channel = [params.REGISTERCHANNELS_SC]
    descriptor_output_dir_fixed = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_%s/',params.SAMPLE_NAME, ...
        fixed_run,register_channel{1}));

    files = dir(fullfile(descriptor_output_dir_fixed,'*.mat'));

    for file_idx= 1:length(files)
        filename = files(file_idx).name;

        %The data for each tile is keys, xmin, xmax, ymin, ymax
        data = load(fullfile(descriptor_output_dir_fixed,filename));
        keys = vertcat(data.keys{:});
        pos = [[keys(:).y]'+data.ymin-1,[keys(:).x]'+data.xmin-1,[keys(:).z]'];

        keys_fixed_total_sc.pos = vertcat(keys_fixed_total_sc.pos,pos);
    end
end
fprintf('load sc keys of fixed round%03d (mod). ',fixed_run);toc;
%------------All descriptors are now loaded as keys_*_total -------------%


%don't need to worry about padding because these tiles are close enough in
%(x,y) origins
tile_upperleft_y_fixed = floor(linspace(1,imgFixed_total_size(1),params.ROWS_TFORM+1));
tile_upperleft_x_fixed = floor(linspace(1,imgFixed_total_size(2),params.COLS_TFORM+1));


for x_idx=1:params.COLS_TFORM
    for y_idx=1:params.ROWS_TFORM

        disp(['Running on row ' num2str(y_idx) ' and col ' num2str(x_idx) ]);

        %Loading the fixed tiles is detemined by some extra overlap between
        %the tiles (may not be necessary)
        tile_img_fixed_nopadding = imgFixed_total(tile_upperleft_y_fixed(y_idx):tile_upperleft_y_fixed(y_idx+1), ...
            tile_upperleft_x_fixed(x_idx):tile_upperleft_x_fixed(x_idx+1), ...
            :);
        tilesize_fixed = size(tile_img_fixed_nopadding);

        ymin_fixed = tile_upperleft_y_fixed(y_idx);
        ymax_fixed = tile_upperleft_y_fixed(y_idx+1);
        xmin_fixed = tile_upperleft_x_fixed(x_idx);
        xmax_fixed = tile_upperleft_x_fixed(x_idx+1);

        ymin_fixed_overlap = floor(max(tile_upperleft_y_fixed(y_idx)-(params.OVERLAP/2)*tilesize_fixed(1),1));
        ymax_fixed_overlap = floor(min(tile_upperleft_y_fixed(y_idx+1)+(params.OVERLAP/2)*tilesize_fixed(1),imgFixed_total_size(1)));
        xmin_fixed_overlap = floor(max(tile_upperleft_x_fixed(x_idx)-(params.OVERLAP/2)*tilesize_fixed(2),1));
        xmax_fixed_overlap = floor(min(tile_upperleft_x_fixed(x_idx+1)+(params.OVERLAP/2)*tilesize_fixed(2),imgFixed_total_size(2)));

        clear tile_img_fixed_nopadding;

        tile_img_fixed = imgFixed_total(ymin_fixed_overlap:ymax_fixed_overlap, xmin_fixed_overlap:xmax_fixed_overlap,:);

        if checkIfTileEmpty(tile_img_fixed,params.EMPTY_TILE_THRESHOLD)
            disp('Sees the moving tile to be empty');
            continue
        end

        clear tile_img_fixed;

        %FindRelevant keys not only finds the total keypoints, but converts
        %those keypoints to the scope of the specific tile, not the global
        %position

        tic;
        keys_fixed_sift_index = find(keys_fixed_total_sift.pos(:,1)>=ymin_fixed & keys_fixed_total_sift.pos(:,1)<=ymax_fixed & ...
            keys_fixed_total_sift.pos(:,2)>=xmin_fixed & keys_fixed_total_sift.pos(:,2)<=xmax_fixed);
        keys_fixed_sift.pos = keys_fixed_total_sift.pos(keys_fixed_sift_index,:)-[ymin_fixed-1,xmin_fixed-1,0];
        keys_fixed_sift.ivec = keys_fixed_total_sift.ivec(keys_fixed_sift_index,:);
        keys_fixed_sc_index = find(keys_fixed_total_sc.pos(:,1)>=ymin_fixed & keys_fixed_total_sc.pos(:,1)<=ymax_fixed & ...
            keys_fixed_total_sc.pos(:,2)>=xmin_fixed & keys_fixed_total_sc.pos(:,2)<=xmax_fixed);
        keys_fixed_sc.pos = keys_fixed_total_sc.pos(keys_fixed_sc_index,:)-[ymin_fixed-1,xmin_fixed-1,0];
        fprintf('findRelevantKeys. keys_fixed(mod) ');toc;

        num_keys_fixed = length(keys_fixed_sift)+length(keys_fixed_sc);
        disp(['Sees ' num2str(num_keys_fixed) ' features for fixed']);
        if num_keys_fixed==0
            disp('Empty set of descriptors. Skipping')
            continue;
        end

        % ----------- SIFT MATCHING AND ROBUST MODEL SELECTION ----------%
        %

        %Extract the keypoints-only for the shape context calculation
        %F for fixed
        DF_SIFT = keys_fixed_sift.ivec;
        LF_SIFT = keys_fixed_sift.pos;
        LF_SC = keys_fixed_sc.pos;
        fprintf('prepare keypoints of fixed round.');toc;

        % deuplicate any SIFT keypoints
        fprintf('(%i) before dedupe, ',size(LF_SIFT,1));
        [LF_SIFT,keepF,~] = unique(LF_SIFT,'rows');
        DF_SIFT = DF_SIFT(keepF,:);
        fprintf('(%i) after dedupe\n',size(LF_SIFT,1));

        % deuplicate any ShapeContext keypoints
        fprintf('(%i) before dedupe, ',size(LF_SC,1));
        [LF_SC,~,~] = unique(LF_SC,'rows');
        fprintf('(%i) after dedupe\n',size(LF_SC,1));


        fprintf('normalizing SIFT descriptors...\n');
        tic;
        DF_SIFT = double(DF_SIFT);
        DF_SIFT_norm = DF_SIFT ./ repmat(sum(DF_SIFT,2),1,size(DF_SIFT,2));
        clear DF_SIFT;
        size(DF_SIFT_norm)
        toc;

        fprintf('calculating ShapeContext descriptors...\n');
        tic;
        %We create a shape context descriptor for the same keypoint
        %that has the SIFT descriptor.
        %So we calculate the SIFT descriptor on the normed channel
        %(summedNorm), and we calculate the Shape Context descriptor
        %using keypoints from all other channels
        DF_SC=ShapeContext(LF_SIFT,LF_SC);
        toc;

        tic;
        lf_sift_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_lf_sift_r%uc%u.mat',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
%        save(lf_sift_filename,'LF_SIFT','DF_SIFT_norm','DF_SC','imgFixed_total_size','num_keys_fixed','ymin_fixed','xmin_fixed');
        save(lf_sift_filename,'LF_SIFT','imgFixed_total_size','num_keys_fixed','ymin_fixed','xmin_fixed');

        tic;
        df_sift_norm_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_df_sift_norm_r%uc%u.bin',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
        fid = fopen(df_sift_norm_filename,'w');
        DF_SIFT_norm_size1 = size(DF_SIFT_norm,1);
        DF_SIFT_norm_size2 = size(DF_SIFT_norm,2);
        fwrite(fid,DF_SIFT_norm_size1,'integer*4');
        fwrite(fid,DF_SIFT_norm_size2,'integer*4');
        fwrite(fid,DF_SIFT_norm,'double');
        fclose(fid);
        fprintf('save DF_SIFT_norm data ');toc;

        tic;
        df_sc_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_df_sc_r%uc%u.bin',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
        fid = fopen(df_sc_filename,'w');
        DF_SC_size1 = size(DF_SC,1);
        DF_SC_size2 = size(DF_SC,2);
        fwrite(fid,DF_SC_size2,'integer*4');
        fwrite(fid,DF_SC_size1,'integer*4');
        fwrite(fid,DF_SC','double');
        fclose(fid);
        fprintf('save DF_SC data ');toc;

        tic;
        df_sift_sc_filename = fullfile(params.OUTPUTDIR,sprintf('%sround%03d_df_sift_sc_r%uc%u.bin',...
            params.SAMPLE_NAME,fixed_run,y_idx,x_idx));
        fid = fopen(df_sift_sc_filename,'w');
        fwrite(fid,DF_SIFT_norm_size1,'integer*4');
        fwrite(fid,DF_SIFT_norm_size2+DF_SC_size1,'integer*4');
        fwrite(fid,[DF_SC; DF_SIFT_norm']','double');
        fclose(fid);
        fprintf('save DF_SC+DF_SIFT_norm data ');toc;
    end
end

