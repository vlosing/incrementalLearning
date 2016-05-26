function [x] = splitArrayBySize(array, splitSize)
    if size(array,1) == 1
        splitDimension = 2;
    else
        splitDimension = 1;
    end
    arraysize = fix(size(array, splitDimension));
    idx=1;
    x = {};
    while idx<=arraysize
        if splitDimension == 2
            to = min(idx+splitSize-1, size(array,2));
            x{end + 1} = array(idx:to);         
        else
            to = min(idx+splitSize-1, size(array,1));
            x{end + 1} = array(idx:to,:);
        end
        idx = idx + splitSize;
    end
end