function predictedTrainLabels = alternateFitPredictISVM(features, labels, kernel, CVar, KSCALE, maxReserveVectors)
labels = labels';
if strcmp(kernel, 'linear') == 1
    kernelType = 1;
elseif strcmp(kernel, 'RBF') == 1
    kernelType = 5;
else
    disp(['unknown kernel ', kernel]);
end
KSCALE = double(KSCALE);
CVar = double(CVar);
global models;
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
max_reserve_vectors = maxReserveVectors;
predictedTrainLabels = -1 * ones(length(labels),1);
for i=1:length(labels)
    if ~allModelsEmpty()
        predictedTrainLabels(i) = predictISVM(features(:,i)); 
    end
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
            svmtrain(features(:,i),localLabel, CVar, kernelType, KSCALE);
            models{classIdx} = updateModel(models{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
        elseif (~models{classIdx}.isEmpty)
            modelToParams(models{classIdx});
            svmtrain(features(:,i),localLabel, CVar);
            %save svm model (maybe not necessary after initial set...depends whether deep copy or not)
            models{classIdx} = updateModel(models{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
        end
    end
end

end
