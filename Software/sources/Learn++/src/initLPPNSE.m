function initLPPNSE(classes, sigmoidSlope, sigmoidCutOff, errorThreshold)
    global net;
    net.beta = [];        % beta will set the classifier weights
    model.type = 'CART';
    net.base_classifier = model;
    net.classes = sort(classes);    
    net.classifiers = {};
    net.initialized=0;
    net.classifierweigths = {};
    net.t = 1;            % track the time of learning
    net.a = sigmoidSlope;                   % slope parameter to a sigmoid
    net.b = double(sigmoidCutOff);          % cutoff parameter to a sigmoid
    net.threshold = errorThreshold;         % how small is too small for error  
end

