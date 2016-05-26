function [net] = initIELM(trainFeatures, trainLabels, activationFunction, nHiddenNeurons)
labels = sort(unique(trainLabels));
nClass=length(labels);

net.nInputNeurons = size(trainFeatures, 2);
net.nClass = nClass;
net.labels = labels;
net.initialized = 0;
net.activationFunction = activationFunction;
net.nHiddenNeurons = nHiddenNeurons;
net.IW = rand(net.nHiddenNeurons,net.nInputNeurons)*2-1;



end

