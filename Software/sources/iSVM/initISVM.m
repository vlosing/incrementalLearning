function initISVM(classes)
global models;
numOfClasses = length(classes);
if numOfClasses > 2
    models   = cell(1,numOfClasses);
    for i=1:numOfClasses
        models{i}.isEmpty = true;
    end
else
    models   = cell(1,1);
    models{1}.isEmpty = true;
end