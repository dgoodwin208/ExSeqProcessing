function index_pairs = match_3DSIFTdescriptors(desc1,desc2)

    sq_threshold = 1.5;

    dist2 = bsxfun(@plus,sum(desc1.^2,2),sum(desc2.^2,2)') - 2*(desc1*desc2');

    [first,idx_f] = min(dist2,[],2);
    dist2(sub2ind(size(dist2),1:size(dist2,1),idx_f')) = Inf;
    [second,idx_s] = min(dist2,[],2);

    passed_index = find(first*sq_threshold < second);
    index_pairs = [passed_index,idx_f(passed_index)]';

end

