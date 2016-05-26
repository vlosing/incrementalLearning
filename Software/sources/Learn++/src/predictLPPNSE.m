function [labels] = predictLPPNSE(data)
    global net;
    [labels, posterior] = classify_ensemble(net, data, net.w(end,:));
end

