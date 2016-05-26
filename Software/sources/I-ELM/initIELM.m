function initIELM(classes, nInputNeurons, activationFunction, nHiddenNeurons)
rng('shuffle');
global net;
labels = sort(unique(classes));
nClass=length(labels);

net.nInputNeurons = nInputNeurons;
net.nClass = nClass;
net.labels = labels;
net.initialized = 0;
net.activationFunction = activationFunction;
net.nHiddenNeurons = nHiddenNeurons;
net.IW = rand(net.nHiddenNeurons,net.nInputNeurons)*2-1;

end

