function label_indices = predictIELM(features)
global net;
switch lower(net.activationFunction)
    case{'rbf'}
        HTest = RBFun(features, net.IW, net.Bias);
    case{'sig'}
        HTest = SigActFun(features, net.IW, net.Bias);
    case{'sin'}
        HTest = SinActFun(features, net.IW, net.Bias);
    case{'hardlim'}
        HTest = HardlimActFun(features, net.IW, net.Bias);
end
Y=HTest * net.beta;

[x, label_indices]=max(Y,[],2);
label_indices=label_indices-1;

end

