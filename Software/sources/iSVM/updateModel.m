function newModel = updateModel(model, a, b, C, deps, g, ind, kernel_evals, perturbations,  Q,  Rs, scale, type, uind, X, y)
    newModel = model;
    newModel.a  = a;
    newModel.b  = b;
    newModel.C  = C;
    newModel.deps  = deps;
    newModel.g  = g;
    newModel.ind  = ind;
    newModel.kernel_evals  = kernel_evals;
    newModel.perturbations  = perturbations;
    newModel.Q  = Q;
    newModel.Rs  = Rs;
    newModel.scale  = scale;
    newModel.type  = type;
    newModel.uind  = uind;
    newModel.X  = X;
    newModel.y  = y;
    
    

