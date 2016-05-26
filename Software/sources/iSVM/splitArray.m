function [x] = splitArray(array, parts)
    if size(array,1) == 1
        splitDimension = 2;
    else
        splitDimension = 1;
    end
    arraysize = fix(size(array, splitDimension));
    partsize = fix(arraysize/parts);
    modulo = mod(arraysize, parts);
    
    x = cell(1,parts);
    addNext = 0;
    for part=1:parts
        from = (part-1)*partsize+1 + addNext;
        to = from+partsize-1;
        if modulo > 0
            to = to +1;
            modulo = modulo - 1;
            addNext = addNext + 1;
        end
        if splitDimension == 2
            x{part} = array(from:to);
        else
            x{part} = array(from:to,:);
        end
    end
end

