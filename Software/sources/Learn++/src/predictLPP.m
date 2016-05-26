function [labels] = predictLPP(data)
    global net;
    [labels, posterior] = classify_ensemble(net, data, log(1./net.beta(1:length(net.classifiers))));
end

