function fitISVM(features, labels, kernel, CVar, KSCALE, maxReserveVectors)
% flags for example state
MARGIN    = 1;
ERROR     = 2;
labels = labels';
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
if strcmp(kernel, 'linear') == 1
    kernelType = 1;
elseif strcmp(kernel, 'RBF') == 1
    kernelType = 5;
else
    disp(['unknown kernel ', kernel]);
end
KSCALE = double(KSCALE);
CVar = double(CVar);
numMarginVectors = 0;
numErrorVectors = 0;
for classIdx=1:size(models, 2)
    localLabels=labels;
    localLabels(localLabels~=classIdx-1)=-1;
    localLabels(localLabels==classIdx-1)=1;
    classIndices = find(localLabels==1);
    sameLabelSamples = numel(classIndices);
    if (models{classIdx}.isEmpty)
        if sameLabelSamples > 0
            models{classIdx}.isEmpty = false;
            svmtrain(features(:,classIndices(1):end),localLabels(classIndices(1):end), CVar, kernelType, KSCALE);
            models{classIdx} = updateModel(models{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
        else
            continue;
        end
    else
        modelToParams(models{classIdx});
        svmtrain(features,localLabels, CVar);
        %save svm model (maybe not necessary after initial set...depends whether deep copy or not)
        models{classIdx} = updateModel(models{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
    end
    numMarginVectors = numMarginVectors + length(models{classIdx}.ind{MARGIN});
    numErrorVectors = numErrorVectors + length(models{classIdx}.ind{ERROR});

end
end
