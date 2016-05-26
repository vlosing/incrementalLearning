function initLearnPP(numOfClassifiersPerChunk, classes)
rng('shuffle');
global net;
net.beta = [];        % beta will set the classifier weights
model.type = 'CART';
net.base_classifier = model;
net.iterations = numOfClassifiersPerChunk;
net.classes = sort(classes);
net.classifiers = {};
net.initialized=0;
net.classifierweigths = {};
end

