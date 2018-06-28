
function is_same = compare_mat_files(srcDir, dstDir)

    srcFiles = dir(fullfile(srcDir,'**','*.mat'));
    dstFiles = dir(fullfile(dstDir,'**','*.mat'));

    if length(srcFiles) ~= length(dstFiles)
        disp('NG - # of files is different.');
        srcFiles
        dstFiles
        is_same = false;
        return;
    end

    is_same = true;
    for i=1:length(srcFiles)
        m1=load(fullfile(srcFiles(i).folder,srcFiles(i).name));
        m2=load(fullfile(dstFiles(i).folder,dstFiles(i).name));

        if isequal(m1,m2)
            disp(['OK - ',srcFiles(i).name])
        else
            %disp(['CHECK FIELDS - ',srcFiles(i).name])
            is_same_field = compare_fields(m1,m2);
            if is_same_field
                disp(['OK - ',srcFiles(i).name])
            else
                disp(['NG - ',srcFiles(i).name])
                is_same = false;
            end
        end
    end

    if is_same
        disp('Total: OK')
    else
        disp('Total: NG')
    end

end

function is_same = compare_fields(obj1, obj2)

    if isstruct(obj1)
        if ~isstruct(obj2)
            disp('  - object1(struct) is a different type from object2.')
            is_same = false;
            return;
        end
        if length(fieldnames(obj1)) ~= length(fieldnames(obj2))
            disp('  - # of object1(struct) fieldnames is different from object2(struct) ones.')
            is_same = false;
            return;
        end

        fields = fieldnames(obj1);
        for f_i = 1:length(fields)
            if ~isequal(obj1.(fields{f_i}),obj2.(fields{f_i}))
                %disp(['  - struct field=',fields{f_i},' is different. Check more.'])
                is_same = compare_fields(obj1.(fields{f_i}),obj2.(fields{f_i}));
                if ~is_same
                    return;
                end
            end
        end
    elseif iscell(obj1)
        if ~iscell(obj2)
            disp('  - object1(cell) is a different type from object2.')
            is_same = false;
            return;
        end
        if length(obj1) ~= length(obj2)
            disp('  - # of object1(cell) is different from object2(cell).')
            is_same = false;
            return;
        end

        for c_i = 1:length(obj1)
            if ~isequal(obj1{c_i},obj2{c_i})
                %disp(['  - cell c_i=',num2str(c_i),' is different. Check more.'])
                is_same = compare_fields(obj1{c_i},obj2{c_i});
                if ~is_same
                    return;
                end
            end
        end
    elseif isobject(obj1)
        if ~isobject(obj2)
            disp('  - object1(class) is a different type from object2.')
            is_same = false;
            return;
        end
        if length(properties(obj1)) ~= length(properties(obj2))
            disp('  - # of object1(class) properties is different from object2(class) ones.')
            is_same = false;
            return;
        end

        props = properties(obj1);
        for p_i = 1:length(props)
            if ~isequal(obj1.(props{p_i}),obj2.(props{p_i}))
                %disp(['  - object properties=',props{p_i},' is different. Check more.'])
                is_same = compare_fields(obj1.(props{p_i}),obj2.(props{p_i}));
                if ~is_same
                    return;
                end
            end
        end
    else
        if ischar(obj1) & ischar(obj2)
            r1 = regexp(obj1,'/([\w/]+)+');
            r2 = regexp(obj2,'/([\w/]+)+');
            if length(r1) > 0 & length(r2) > 0
                is_same = true;
            else
                disp('  - object1(char) is not the same as object2(char).')
                disp(['  - ',obj1,'; ',obj2'])
                is_same = false;
            end
        elseif isequal(obj1,obj2)
            is_same = true;
        else
            is_same = false;
        end
    end
end
