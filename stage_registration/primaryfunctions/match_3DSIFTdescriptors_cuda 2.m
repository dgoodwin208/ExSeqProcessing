function index_pairs = match_3DSIFTdescriptors_cuda(desc1,desc2)

    sq_threshold = 1.5;

    [sq_dist,idx] = nnsearch2_cuda(desc1,desc2);

    passed_index = find(sq_dist(:,1)*sq_threshold < sq_dist(:,2));
    index_pairs = [passed_index,idx(passed_index)]';

end

