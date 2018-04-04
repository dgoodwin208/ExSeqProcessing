function index_pairs = match_3DSIFTdescriptors_cuda(desc1,desc2)

    sq_threshold = 1.5;

    [sq_dist,idx] = nnsearch2_cuda(desc1,desc2);

    passed_index = find(sq_dist(:,1)*sq_threshold < sq_dist(:,2));
    index_pairs = [passed_index,idx(passed_index)]';

%    format shortG
%    [passed_index,sq_dist(passed_index,:),sq_dist(passed_index,2)./sq_dist(passed_index,1)]
%
%    passed_index2 = find(sq_dist(:,1)*sq_threshold >= sq_dist(:,2) & sq_dist(:,1)*1.49 < sq_dist(:,2));
%    [passed_index2,sq_dist(passed_index2,:),sq_dist(passed_index2,2)./sq_dist(passed_index2,1)]

end

