function norm = quantilenorm_simple(data)

    row_size = size(data,1);
    col_size = size(data,2);

    sort1_start = tic;
    disp('sort 1');
    d = cell(1,col_size);
    d_mean = zeros(row_size,1);

    div_size = 8;
    sub_idx = [1:ceil(row_size/div_size):row_size row_size+1];

    for i = 1:col_size
        fprintf('col=%d\n',i);

        for j = 1:(length(sub_idx)-1)
            if sub_idx(j) == sub_idx(j+1)-1
                continue
            end

            sort_start = tic;
            g = gpuArray([data(sub_idx(j):(sub_idx(j+1)-1),i) (sub_idx(j):(sub_idx(j+1)-1))']);

            k = ceil(j/2);
            if mod(j,2) == 1
                d{i}{k} = gather(sortrows(g));
            else
                d{i}{k} = mergearrays(d{i}{k},gather(sortrows(g)));
            end
            toc(sort_start)
        end
        d{i} = mergearrays4(d{i}{1},d{i}{2},d{i}{3},d{i}{4});

        d_mean = d_mean+d{i}(:,1);
    end
    d_mean = d_mean./col_size;
    disp('sort 1 total');
    toc(sort1_start)
    %save('quantilenorm_simple_sort1.mat','d','d_mean','-v7.3');
    %load('quantilenorm_simple_sort1.mat','-v7.3');

    sort2_start = tic;
    disp('sort 2');
    norm = zeros(size(data));
    for i = 1:col_size
        fprintf('col=%d\n',i);
        t_start = tic;
        d_diff = diff(d{i}(:,1));
        same_d_idx = find(d_diff==0);
        disp(['same_d_idx=',num2str(length(same_d_idx))])

        mean_start = tic;
        d{i} = circshift(d{i},[0, 1]);
        %d{i}(:,1) = d{i}(:,2);
        %d{i}(:,2) = d_mean;
        if ~isempty(same_d_idx)
            d{i}(:,2) = averageSameRanks(d_mean,same_d_idx);
        end
        toc(mean_start)

        disp('==')

        sort_d = cell(1,div_size/2);
        for j = 1:(length(sub_idx)-1)
            if sub_idx(j) == sub_idx(j+1)-1
                continue
            end

            sort_start = tic;
            g = gpuArray(d{i}(sub_idx(j):(sub_idx(j+1)-1),:));
            k = ceil(j/2);
            if mod(j,2) == 1
                sort_d{k} = gather(sortrows(g));
            else
                sort_d{k} = mergearrays(sort_d{k},gather(sortrows(g)));
            end
            toc(sort_start)
        end
        disp('--')
        m_start = tic;
        sort_d{1} = mergearrays4(sort_d{1},sort_d{2},sort_d{3},sort_d{4});
        toc(m_start)

        norm(:,i) = sort_d{1}(:,2);
        %clear sort_d;

        toc(t_start)
        disp('#####')
    end
    disp('sort 2 total');
    toc(sort2_start)

end

