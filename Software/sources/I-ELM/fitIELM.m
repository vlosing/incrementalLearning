function fitIELM(trainFeatures, trainLabels)
global net;
%%%%%%%%%% Processing the labels
labelMatrix=zeros(length(trainLabels),net.nClass);
for j = 1:net.nClass
   labelMatrix(:,j) = trainLabels == net.labels(j);
end
labelMatrix(labelMatrix==0)=-1;

if ~net.initialized
    %%%%%%%%%%% step 1 Initialization Phase
    switch lower(net.activationFunction)
        case{'rbf'}
            net.Bias = rand(1,net.nHiddenNeurons);
            %        Bias = rand(1,nHiddenNeurons)*1/3+1/11;     %%%%%%%%%%%%% for the cases of Image Segment and Satellite Image
            %        Bias = rand(1,nHiddenNeurons)*1/20+1/60;    %%%%%%%%%%%%% for the case of DNA
            H0 = RBFun(trainFeatures,net.IW,net.Bias);
        case{'sig'}
            net.Bias = rand(1,net.nHiddenNeurons)*2-1;
            H0 = SigActFun(trainFeatures,net.IW,net.Bias);
        case{'sin'}
            net.Bias = rand(1,net.nHiddenNeurons)*2-1;
            H0 = SinActFun(trainFeatures,net.IW,net.Bias);
        case{'hardlim'}
            net.Bias = rand(1,net.nHiddenNeurons)*2-1;
            H0 = HardlimActFun(trainFeatures,net.IW,net.Bias);
            H0 = double(H0);
    end
    net.initialized = 1;
    net.M = pinv(H0' * H0);
    net.beta = pinv(H0) * labelMatrix;
else
    %%%%%%%%%%%%% step 2 Sequential Learning Phase
    switch lower(net.activationFunction)
        case{'rbf'}
            H = RBFun(trainFeatures,net.IW,net.Bias);
        case{'sig'}
            H = SigActFun(trainFeatures,net.IW,net.Bias);
        case{'sin'}
            H = SinActFun(trainFeatures,net.IW,net.Bias);
        case{'hardlim'}
            H = HardlimActFun(trainFeatures,net.IW,net.Bias);
    end
    net.M = net.M - net.M * H' * (eye(size(trainFeatures,1)) + H * net.M * H')^(-1) * H * net.M;
    net.beta = net.beta + net.M * H' * (labelMatrix - H * net.beta);
end
end
