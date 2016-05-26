function [resModels, numMarginVectors, numErrorVectors] = fitISVMSingle(features, labels, models, numOfClasses, KTYPE, CVar, KSCALE, maxReserveVectors)
    % flags for example state
    MARGIN    = 1;
    ERROR     = 2;
    RESERVE   = 3;
    UNLEARNED = 4;
    global a;                     % alpha coefficients
    global b;                     % bias
    global C;                     % regularization parameters 
    global deps;                  % jitter factor in kernel matrix
    global g;                     % partial derivatives of cost function w.r.t. alpha coefficients
    global ind;                   % cell array containing indices of margin, error, reserve and unlearned vectors
    global kernel_evals;          % kernel evaluations
    global max_reserve_vectors;   % maximum number of reserve vectors stored
    global perturbations;         % number of perturbations
    global Q;                     % extended kernel matrix for all vectors
    global Rs;                    % inverse of extended kernel matrix for margin vectors   
    global scale;                 % kernel scale
    global type;                  % kernel type
    global uind;                  % user-defined example indices
    global X;                     % matrix of margin, error, reserve and unlearned vectors stored columnwise
    global y;                     % column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors    
    global counter;
    max_reserve_vectors = maxReserveVectors;
    for i=1:length(labels)
        currentClass = labels(i);
        for classIdx=1:size(models, 2)
            if classIdx == currentClass+1
                localLabel = 1;
            else 
                localLabel = -1;
            end
            if (models{classIdx}.isEmpty) & (localLabel == 1)
                models{classIdx}.isEmpty = false;
                models{classIdx}.offset = i-1;
                svmtrain(features(:,i),localLabel, CVar, KTYPE, KSCALE);        
                models{classIdx} = updateModel(models{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
            elseif (~models{classIdx}.isEmpty)
                modelToParams(models{classIdx});
                svmtrain(features(:,i),localLabel, CVar);        
                %save svm model (maybe not necessary after initial set...depends whether deep copy or not)
                models{classIdx} = updateModel(models{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
            end
        end  
    end
    numMarginVectors = 0;
    numErrorVectors = 0;    
    for classIdx=1:size(models, 2)
        if ~(models{classIdx}.isEmpty)
            numMarginVectors = numMarginVectors + length(models{classIdx}.ind{MARGIN});
            numErrorVectors = numErrorVectors + length(models{classIdx}.ind{ERROR});
        end
    end
    resModels = models;
end
