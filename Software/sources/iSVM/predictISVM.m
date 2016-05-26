function labels = predictISVM(features)
    global models;
    global scale;                 % kernel scale
    global type;                  % kernel type
    margins  = zeros(size(features, 2), size(models, 2));
    for classIdx=1:size(models, 2)
        if ~models{classIdx}.isEmpty
            [margins(:, classIdx)] = svmeval(features, models{classIdx}.a,models{classIdx}.b,models{classIdx}.ind,models{classIdx}.X,models{classIdx}.y,type,scale);                
        else
            %disp(['no train samples for class ', num2str(classIdx-1)]);
            [margins(:, classIdx)] = ones(size(features, 2),1)*-9999;
        end
    end
    if size(models, 2) == 1
       labels = margins < 0;
    else
        [dummy,labels]  = max(margins, [], 2);
        labels = labels - 1;
     
end
