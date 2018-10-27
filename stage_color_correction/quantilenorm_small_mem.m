function [varargout] = quantilenorm_small_mem(outputdir,basename,varargin)

    if nargin < 3
        disp('input args < 3')
        return;
    end

    col_size = nargin-2;

    image_info{1} = imfinfo(varargin{1});
    num_images = numel(image_info{1});
    if num_images < 1
        disp('input image has no image')
        return;
    end

    image_width  = image_info{1}(1).Width;
    image_height = image_info{1}(1).Height;

    %div_size = 32;
    div_size = 16;
    %div_size = 8;

    sort1_start = tic;
    disp('#####  sort 1');

    sub_idx = ceil(linspace(1,num_images+1,div_size+1));
    if sum(diff(sub_idx)) ~= num_images
        disp('sub_idx list is wrong')
        return
    end

    % prepare file list at each phase
    makelist1_start = tic;
    disp('## make lists of mergefiles1 and sortedfiles1')
    merge1file_list = {};
    m_i = 1;
    for i = 1:col_size
        fname = varargin{i};
        image_info{i} = imfinfo(fname);
        prefix = sprintf('%s_1_sort1_c%d',basename,i);

        s = [sub_idx(1:end-1);sub_idx(2:end)-1]';
        while size(s,1) > 1
            t = [s(1:2:end-1,:) s(2:2:end,:)];
            u = t(:,[1 4]);
            for j = 1:size(t,1)
                in1_file = sprintf('%s-%03d-%03d.bin',prefix,t(j,1),t(j,2));
                in2_file = sprintf('%s-%03d-%03d.bin',prefix,t(j,3),t(j,4));
                out_file = sprintf('%s-%03d-%03d.bin',prefix,t(j,1),t(j,4));

                merge1file_list{m_i} = { in1_file,in2_file,out_file };
                m_i = m_i+1;
            end

            s = u;
        end
    end

    sort1file_list = {};
    for i = 1:col_size
        sort1file_list{i} = sprintf('%s_1_sort1_c%d-%03d-%03d.bin',basename,i,sub_idx(1),sub_idx(end)-1);
    end
    toc(makelist1_start)


    % execute phases
    sort1_sub_start = tic;
    disp('## sort1 sub-tasks')

    sort1files_exist = true;
    for i = 1:col_size
        if ~exist(fullfile(outputdir,sort1file_list{i}))
            sort1files_exist = false;
            break;
        end
    end

    if ~sort1files_exist
        f_merge = parfeval(@mergesortfiles,0,outputdir,merge1file_list);
        disp('# start merge sort')
    
        for i = 1:(div_size*col_size)
            col_i = ceil(i/div_size);
            sub_i = i-div_size*(col_i-1);

            fname = varargin{col_i};
    
            fprintf('# start sort1 (%d)\n',i)
            try
            f_sort1(i) = parfeval(@sort1,0,outputdir,basename,fname,image_info{col_i},col_i,sub_i,sub_idx,image_width,image_height);
            %sort1(outputdir,basename,fname,image_info{col_i},col_i,sub_i,sub_idx,image_width,image_height);
            catch ME
                disp(ME.getReport)
            end
        end
    
        try
        disp('# wait sort1')
        fetchOutputs(f_sort1);
        catch ME
            disp(ME.getReport)
        end
        disp('# wait merge')
        fetchOutputs(f_merge);
        for i = 1:(div_size*col_size)
            f_sort1(i).Diary
        end
        f_merge.Diary
        %mergefiles(outputdir,merge1file_list);
    end
    toc(sort1_sub_start)

    mean_start = tic;
    disp('## mean');
    mean_file = sprintf('%s_2_sort1_mean.bin',basename);
    if ~exist(fullfile(outputdir,mean_file))
        meanfiles(outputdir,sort1file_list,mean_file);
    end
    toc(mean_start)

    disp('##### sort 1 total');
    toc(sort1_start)


    sort2_start = tic;
    disp('##### sort 2');

    substitute_to_norm_start = tic;
    disp('## substitute to normalized mean values');

    substfile_list = {};
    for i = 1:col_size
        substfile_list{i} = sprintf('%s_3_subst_c%d.bin',basename,i);
    end

    substfiles_exist = true;
    for i = 1:col_size
        if ~exist(fullfile(outputdir,substfile_list{i}))
            substfiles_exist = false;
            break;
        end
    end

    if ~substfiles_exist
        substituteToNormValues(outputdir,mean_file,sort1file_list,substfile_list);
    end
    toc(substitute_to_norm_start)


    makelist2_start = tic;
    disp('## make lists of mergefile2')
    merge2file_list = {};
    m_i = 1;
    for i = 1:col_size
        prefix = sprintf('%s_4_sort2_c%d',basename,i);

        s = [sub_idx(1:end-1);sub_idx(2:end)-1]';
        while size(s,1) > 1
            t = [s(1:2:end-1,:) s(2:2:end,:)];
            u = t(:,[1 4]);
            for j = 1:size(t,1)
                in1_file = sprintf('%s-%03d-%03d.bin',prefix,t(j,1),t(j,2));
                in2_file = sprintf('%s-%03d-%03d.bin',prefix,t(j,3),t(j,4));
                out_file = sprintf('%s-%03d-%03d.bin',prefix,t(j,1),t(j,4));

                merge2file_list{m_i} = { in1_file,in2_file,out_file };
                m_i = m_i+1;
            end

            s = u;
        end
    end
    toc(makelist2_start)


    sort2_sub_start = tic;
    disp('## sort2 sub-tasks')

    sort2files_exist = true;
    for i = 1:col_size
        sortedfile = sprintf('%s_4_sort2_c%d-%03d-%03d.bin',basename,i,sub_idx(1),sub_idx(end)-1);
        if ~exist(fullfile(outputdir,sortedfile))
            sort2files_exist = false;
            break;
        end
    end

    if ~sort2files_exist
        f_merge = parfeval(@mergesortfiles,0,outputdir,merge2file_list);

        for i = 1:(div_size*col_size)
            col_i = ceil(i/div_size);
            sub_i = i-div_size*(col_i-1);

            fprintf('# start sort2 (%d)\n',i)
            try
            f_sort2(i) = parfeval(@sort2,0,outputdir,basename,substfile_list{col_i},col_i,sub_i,sub_idx,image_width,image_height);
            catch ME
                disp(ME.getReport)
            end
        end

        try
        disp('# wait sort2')
        fetchOutputs(f_sort2);
        catch ME
            disp(ME.getReport)
        end
        disp('# wait merge')
        fetchOutputs(f_merge);
        for i = 1:(div_size*col_size)
            f_sort2(i).Diary
        end
        f_merge.Diary
    end
    toc(sort2_sub_start)

    disp('##### sort 2 total');
    toc(sort2_start)

    for i = 1:col_size
        varargout{i} = sprintf('%s_4_sort2_c%d-%03d-%03d.bin',basename,i,sub_idx(1),sub_idx(end)-1);
    end
    varargout{col_size+1} = image_height;
    varargout{col_size+2} = image_width;

end

% =================================================================================================
function save_mat(filename,mat)
    save(filename,'mat','-v7.3');
end

function save_bin(filename,data)
    fid = fopen(filename,'w');
    if fid == -1
        disp('save_bin: cannot open file')
        return
    end

    fwrite(fid,data,'double');
    fclose(fid);
end

function idx_gpu = selectGPU()
    sem_name = sprintf('/%s.g',getenv('USER'));
    idx_gpu = 0;
    for i = 1:gpuDeviceCount
        ret = semaphore([sem_name num2str(i)],'trywait');
        if ret == 0
            idx_gpu = i;
            break
        end
    end
end

function unselectGPU(idx_gpu)
    sem_name = sprintf('/%s.g%d',getenv('USER'),idx_gpu);
    ret = semaphore(sem_name,'post');
    if ret == -1
        fprintf('unselect [%s] failed.\n',sem_name);
    end
end

function ret = selectCore(num_core_sem)
    sem_name = sprintf('/%s.c%d',getenv('USER'),num_core_num);
    count = 1;
    while true
        ret = semaphore(sem_name,'trywait');
        if ret == 0
            fprintf('selectCore[%s count=%d]\n',sem_name,count);
            break
        end
        count = count + 1;
        pause(2);
    end
end

function ret = selectCoreNoblock(num_core_sem)
    sem_name = sprintf('/%s.c%d',getenv('USER'),num_core_num);
    ret = semaphore(sem_name,'trywait');
    if ret == 0
        fprintf('selectCoreNoblock[%s]\n',sem_name);
    end
end

function unselectCore(num_core_sem)
    sem_name = sprintf('/%s.c%d',getenv('USER'),num_core_num);
    ret = semaphore(sem_name,'post');
    if ret == -1
        fprintf('unselect [%s] failed.\n',sem_name);
    end
end

function sort1(outputdir,basename,fname,image_info,col_i,sub_i,sub_idx,image_width,image_height)

    start_sub_idx = sub_idx(sub_i);
    end_sub_idx   = sub_idx(sub_i+1);

    output_fname     = fullfile(outputdir,sprintf('%s_1_sort1_c%d-%03d-%03d.bin',basename,col_i,start_sub_idx,end_sub_idx-1));
    tmp_output_fname = fullfile(outputdir,sprintf('.tmp.%s_1_sort1_c%d-%03d-%03d.bin',basename,col_i,start_sub_idx,end_sub_idx-1));
    if exist(output_fname,'file')
        delete(output_fname);
    end
    if exist(tmp_output_fname,'file')
        delete(tmp_output_fname);
    end
    system(sprintf('touch %s',tmp_output_fname));

    selectCore(1);

    fpos_start = (start_sub_idx-1)*image_width*image_height+1;
    fpos_end   = (end_sub_idx-1)*image_width*image_height;
    fprintf('## (%d %d; %d %d; %d %d)\n',col_i,sub_i,start_sub_idx,end_sub_idx-1,fpos_start,fpos_end)
    sub_images = zeros([image_width,image_height,end_sub_idx-start_sub_idx]);
    for j = start_sub_idx:(end_sub_idx-1)
        sub_images(:,:,j-start_sub_idx+1) = imread(fname,j,'Info',image_info);
    end

    while true
        idx_gpu = selectGPU();
        if idx_gpu > 0
            gdv = gpuDevice(idx_gpu);
            g = gpuArray([sub_images(:) (fpos_start:fpos_end)']);
            sub_images = [];
            sorted = gather(sortrows(g));
            %sorted = radixsort(sub_images(:),(fpos_start:fpos_end)');
            gpuDevice([]);
            unselectGPU(idx_gpu);
            break;
        else
            ret = selectCoreNoblock(2);
            if ret == 0
                sorted = sortrows([sub_images(:) (fpos_start:fpos_end)']);
                sub_images = [];
                unselectCore(2);
                break;
            else
                pause(1);
            end
        end
    end % while

    unselectCore(1);

    save_bin(tmp_output_fname,sorted');
    movefile(tmp_output_fname,output_fname);

end

function sort2(outputdir,basename,infile,col_i,sub_i,sub_idx,image_width,image_height)

    start_sub_idx = sub_idx(sub_i);
    end_sub_idx   = sub_idx(sub_i+1);

    output_fname     = fullfile(outputdir,sprintf('%s_4_sort2_c%d-%03d-%03d.bin',basename,col_i,start_sub_idx,end_sub_idx-1));
    tmp_output_fname = fullfile(outputdir,sprintf('.tmp.%s_4_sort2_c%d-%03d-%03d.bin',basename,col_i,start_sub_idx,end_sub_idx-1));
    if exist(output_fname,'file')
        delete(output_fname);
    end
    if exist(tmp_output_fname,'file')
        delete(tmp_output_fname);
    end
    system(sprintf('touch %s',tmp_output_fname));

    selectCore(1);

    fpos_start = (start_sub_idx-1)*image_width*image_height;
    fpos_end   = (end_sub_idx-1)*image_width*image_height;
    fprintf('## (%d %d; %d %d; %d %d = %d)\n',col_i,sub_i,start_sub_idx,end_sub_idx-1,fpos_start,fpos_end-1,fpos_end-fpos_start)

    fid = fopen(fullfile(outputdir,infile),'r');
    fseek(fid,fpos_start*2*8,'bof');
    sub_images = fread(fid,[2,fpos_end-fpos_start],'double');
    fclose(fid);

    while true
        idx_gpu = selectGPU();
        if idx_gpu > 0
            gdv = gpuDevice(idx_gpu);
            g = gpuArray(sub_images');
            sub_images = [];
            sorted = gather(sortrows(g));
            %sorted = radixsort(sub_images(1,:)',sub_images(2,:)');
            gpuDevice([]);
            unselectGPU(idx_gpu);
            break;
        else
            ret = selectCoreNoblock(2);
            if ret == 0
                sorted = sortrows(sub_images');
                sub_images = [];
                unselectCore(2);
                break;
            else
                pause(1);
            end
        end
    end % while

    unselectCore(1);

    save_bin(tmp_output_fname,sorted');
    movefile(tmp_output_fname,output_fname);

end

