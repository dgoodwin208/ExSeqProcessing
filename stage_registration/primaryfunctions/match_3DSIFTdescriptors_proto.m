function index_pairs = match_3DSIFTdescriptors_proto(desc1,desc2)

    sq_threshold = 1.5;

    % ###################################################################################
    %[dist,index] = pdist2(desc2,desc1,'squaredeuclidean','Smallest',2);
    %passed_index = find(dist(1,:) * sq_threshold < dist(2,:));
    %index_pairs = [passed_index; index(1,passed_index)];

    % ###################################################################################
    %dist2 = bsxfun(@plus,sum(desc1.^2,2),sum(desc2.^2,2)') - 2*(desc1*desc2');
    %[S,I] = sort(dist2,2);
    %passed_index = find(S(:,1)*sq_threshold < S(:,2));
    %index_pairs = [passed_index,I(passed_index,1)]';

    % ###################################################################################
    %profile on;
    dist2 = bsxfun(@plus,sum(desc1.^2,2),sum(desc2.^2,2)') - 2*(desc1*desc2');

%    f=fopen('sift_dist2_dim.bin','w');
%    fwrite(f,size(dist2,1),'integer*4');
%    fwrite(f,size(dist2,2),'integer*4');
%    fclose(f);
%
%    f=fopen('sift_dist2.bin','w');
%    fwrite(f,dist2,'double');
%    fclose(f);
%    format longG
%    dist2(1,1)
%    dist2(2,1)
%    dist2(3,1)

    [first,idx_f] = min(dist2,[],2);
    dist2(sub2ind(size(dist2),1:size(dist2,1),idx_f')) = Inf;
    [second,idx_s] = min(dist2,[],2);

    passed_index = find(first*sq_threshold < second);
    index_pairs = [passed_index,idx_f(passed_index)]';

%    format shortG
%    [passed_index,first(passed_index),second(passed_index),second(passed_index)./first(passed_index)]
%
%    passed_index2 = find(first*sq_threshold >= second & first*1.49 < second);
%    [passed_index2,first(passed_index2),second(passed_index2),second(passed_index2)./first(passed_index2)]

    %profile off;
    %profsave(profile('info'),sprintf('profile-match-3DSIFTdescriptors-%d',i));

    % ###################################################################################
%    tic;
%    num_chunks1 = 2;
%    num_chunks2 = 2;
%    chunk1_size = ceil(size(desc1,1) / num_chunks1);
%    chunk2_size = ceil(size(desc2,1) / num_chunks2);
%    desc1_n_size = size(desc1,1);
%    desc2_n_size = size(desc2,1);
%    desc_cell = {};
%    toc;
%
%    tic;
%    count = 1;
%    for i1 = 1:num_chunks1
%        i1_start = (i1-1)*chunk1_size+1;
%        i1_end = min(i1*chunk1_size,desc1_n_size);
%        for i2 = 1:num_chunks2
%            i2_start = (i2-1)*chunk2_size+1;
%            i2_end = min(i2*chunk2_size,desc2_n_size);
%
%            desc_cell{count} = {};
%            desc_cell{count}.d1 = desc1(i1_start:i1_end,:);
%            desc_cell{count}.d2 = desc2(i2_start:i2_end,:);
%            desc_cell{count}.i1_start = i1_start;
%            desc_cell{count}.i1_end = i1_end;
%            desc_cell{count}.i2_start = i2_start;
%            desc_cell{count}.i2_end = i2_end;
%
%            %[i1,i2]
%            %desc_cell{count}.d1
%            %desc_cell{count}.d2
%            %desc_cell{count}.i1
%            %desc_cell{count}.i2
%            count = count + 1;
%        end
%    end
%    toc;
%
% %   profile on;
%    tic;
%    ticBytes(gcp)
%    two_tops_cell = {};
%    parfor i = 1:(num_chunks1*num_chunks2)
%        dist2 = bsxfun(@plus,sum(desc_cell{i}.d1.^2,2),sum(desc_cell{i}.d2.^2,2)') - 2*(desc_cell{i}.d1*desc_cell{i}.d2');
%
%        [first,idx_f] = min(dist2,[],2);
%        dist2(sub2ind(size(dist2),1:size(dist2,1),idx_f')) = Inf;
%        [second,idx_s] = min(dist2,[],2);
%
%        two_tops_cell{i} = {};
%        %two_tops_cell{i}{i1_start} = [first,idx_f+(desc_cell{i}.i1_start-1),second,idx_s+(desc_cell{i}.i2_start-1)];
%        two_tops_cell{i}.dist2_min = [first,second]';
%        two_tops_cell{i}.idx = [idx_f+(desc_cell{i}.i1_start-1),idx_s+(desc_cell{i}.i2_start-1)]';
%        two_tops_cell{i}.i1_start = desc_cell{i}.i1_start;
%        two_tops_cell{i}.i1_end = desc_cell{i}.i1_end;
%    end
%    tocBytes(gcp)
%    toc;
%
%    tic;
%    dist2_all = Inf(2,desc1_n_size);
%    idx_all = Inf(2,desc1_n_size);
%
%    for i = 1:(num_chunks1*num_chunks2)
%        i1_start = two_tops_cell{i}.i1_start;
%        i1_end = two_tops_cell{i}.i1_end;
%        i1_size = i1_end-i1_start+1;
%
%        dist2 = zeros(4,i1_size);
%        dist2(1:2,:) = dist2_all(:,i1_start:i1_end);
%        dist2(3:4,:) = two_tops_cell{i}.dist2_min;
%        idx = zeros(4,i1_size);
%        idx(1:2,:) = idx_all(:,i1_start:i1_end);
%        idx(3:4,:) = two_tops_cell{i}.idx;
%
%        [first,idx_f] = min(dist2);
%        dist2(sub2ind(size(dist2),idx_f,1:i1_size)) = Inf;
%        [second,idx_s] = min(dist2);
%
%%        dist2_all
%%        idx_all
%%        two_tops_cell{i}.dist2_min
%%        two_tops_cell{i}.idx
%%        first
%%        second
%
%        dist2_all(1,i1_start:i1_end) = first;
%        dist2_all(2,i1_start:i1_end) = second;
%        idx_all(1,i1_start:i1_end) = idx(sub2ind(size(idx),idx_f,1:i1_size));
%        idx_all(2,i1_start:i1_end) = idx(sub2ind(size(idx),idx_s,1:i1_size));
%
%%        dist2_all
%%        idx_all
%    end
%
%    passed_index = find(dist2_all(1,:)*sq_threshold < dist2_all(2,:));
%    index_pairs = [passed_index;idx_all(passed_index)];
%
%    toc;
%
% %   profile off;
% %   profsave(profile('info'),sprintf('profile-match-3DSIFTdescriptors'));

    % ###################################################################################
%    tic;
%    p = gcp('nocreate');
%    if isempty(p)
%        num_workers = 1;
%    else
%        num_workers = p.NumWorkers;
%    end
%    num_workers
%    %num_workers = 2;
%    chunk_size = ceil(size(desc1,1) / num_workers);
%    desc1_cell = {};
%    toc;
%
%    tic;
%    count = 1;
%    for i = 1:num_workers
%        idx_start = (i-1)*chunk_size+1;
%        idx_end = min(i*chunk_size,size(desc1,1));
%        desc1_cell{count} = desc1(idx_start:idx_end,:);
%
%        %[i,idx_start,idx_end]
%        %desc1_cell{count}
%        count = count + 1;
%    end
%    toc;
%
%    tic;
%    ticBytes(gcp)
%    index_pairs_cell = {};
%    parfor i = 1:num_workers
%        profile on;
%        disp(i)
%        idx_start = (i-1)*chunk_size+1;
%        dist2 = bsxfun(@plus,sum(desc1_cell{i}.^2,2),sum(desc2.^2,2)') - 2*(desc1_cell{i}*desc2');
%
%        [first,idx_f] = min(dist2,[],2);
%        dist2(sub2ind(size(dist2),1:size(dist2,1),idx_f')) = Inf;
%        [second,idx_s] = min(dist2,[],2);
%
%        passed_index = find(first*sq_threshold < second);
%        index_pairs_cell{i} = [passed_index+(idx_start-1),idx_f(passed_index)]';
%
%        profile off;
%        profsave(profile('info'),sprintf('profile-match-3DSIFTdescriptors-%d',i));
%    end
%    tocBytes(gcp)
%    toc;
%
%    tic;
%    index_pairs = [];
%    for i = 1:num_workers
%        index_pairs = horzcat(index_pairs,index_pairs_cell{i});
%    end
%    toc;

    % ###################################################################################
    % use parallel.pool.Constant
    %tic;
    %p = gcp('nocreate');
    %if isempty(p)
    %    num_workers = 1;
    %else
    %    num_workers = p.NumWorkers;
    %end
    %num_workers
    %%num_workers = 2;
    %chunk_size = ceil(size(desc1,1) / num_workers);
    %toc;

    %profile on;
    %desc1_size = size(desc1,1);
    %C = parallel.pool.Constant(desc1);
    %tic;
    %ticBytes(gcp)
    %index_pairs_cell = {};
    %parfor i = 1:num_workers
    %    idx_start = (i-1)*chunk_size+1;
    %    idx_end = min(i*chunk_size,desc1_size);
    %    dist2 = bsxfun(@plus,sum(desc1(idx_start:idx_end,:).^2,2),sum(desc2.^2,2)') - 2*(desc1(idx_start:idx_end,:)*desc2');

    %    [first,idx_f] = min(dist2,[],2);
    %    dist2(sub2ind(size(dist2),1:size(dist2,1),idx_f')) = Inf;
    %    [second,idx_s] = min(dist2,[],2);

    %    passed_index = find(first*sq_threshold < second);
    %    index_pairs_cell{i} = [passed_index+(idx_start-1),idx_f(passed_index)]';
    %end
    %tocBytes(gcp)
    %toc;

    %tic;
    %index_pairs = [];
    %for i = 1:num_workers
    %    index_pairs = horzcat(index_pairs,index_pairs_cell{i});
    %end
    %toc;

    %profile off;
    %profsave(profile('info'),sprintf('profile-match-3DSIFTdescriptors'));
end

