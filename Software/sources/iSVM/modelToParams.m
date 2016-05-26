function modelToParams(model)
    global a;                     % alpha coefficients
    global b;                     % bias
    global C;                     % regularization parameters 
    global deps;                  % jitter factor in kernel matrix
    global g;                     % partial derivatives of cost function w.r.t. alpha coefficients
    global ind;                   % cell array containing indices of margin, error, reserve and unlearned vectors
    global kernel_evals;          % kernel evaluations
    global perturbations;         % number of perturbations
    global Q;                     % extended kernel matrix for all vectors
    global Rs;                    % inverse of extended kernel matrix for margin vectors   
    global scale;                 % kernel scale
    global type;                  % kernel type
    global uind;                  % user-defined example indices
    global X;                     % matrix of margin, error, reserve and unlearned vectors stored columnwise
    global y;                     % column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors

    a = model.a;
    b = model.b;
    C = model.C;
    deps = model.deps;
    g = model.g;
    ind = model.ind;
    kernel_evals = model.kernel_evals;
    perturbations = model.perturbations;
    Q = model.Q;
    Rs = model.Rs;
    scale = model.scale;
    type = model.type;
    uind = model.uind;
    X = model.X;
    y = model.y;


