function rel_path = relpath(src_path, dst_path)

    split_src_path = flip(split(src_path,'/'));
    split_dst_path = flip(split(dst_path,'/'));
    rel_path = {};

    pos = -1;
    for i = 1:min(length(split_src_path),length(split_dst_path))
        if isequal(split_src_path{i},split_dst_path{i}) && ~isempty(split_src_path{i})
            pos = i-1;
            break;
        else
            rel_path{end+1} = split_dst_path{i};
        end
    end

    for i = 1:pos
        rel_path{end+1} = '..';
    end

    rel_path = cell2mat(join(flip(rel_path),'/'));
end

