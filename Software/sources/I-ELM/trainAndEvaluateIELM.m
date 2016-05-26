function trainFromFileAndEvaluateIELM(trainFeatures, trainLabels, testFeatures, evaluationStepSize, evalDstPathPrefix, splitTestfeatures, activationFunction, nHiddenNeurons)
rng('shuffle');
nHiddenNeurons = double(nHiddenNeurons);
%trainFeatures = dlmread(trainFeaturesFileName,'', 1, 0);
%trainLabels = dlmread(trainLabelsFileName,'', 1, 0);
%size(trainFeatures)
%trainFeatures
if size(testFeatures, 1) > 0
    if splitTestfeatures == 1
        testFeaturesChunks = splitArrayBySize(testFeatures, evaluationStepSize);
    end        
end
trainFeaturesChunks = splitArrayBySize(trainFeatures, evaluationStepSize);
trainLabelsChunks = splitArrayBySize(trainLabels, evaluationStepSize);
net = initIELM(trainFeatures, trainLabels, activationFunction, nHiddenNeurons);
numOfChunks = size(trainFeaturesChunks, 2);
for i=1:numOfChunks
    disp(['chunk', num2str(i), '/', num2str(numOfChunks)]);
    if i > 1
        labels = predictIELM(net, trainFeaturesChunks{i});
        fileName = strcat(evalDstPathPrefix, '_');
        fileName = strcat(fileName, num2str(i));
        fileName = strcat(fileName, 'of');
        fileName = strcat(fileName, num2str(numOfChunks));        
        fileName = strcat(fileName, 'predictedTrainLabels.csv');
        dlmwrite(fileName, labels);
    end
    net = fitIELM(net, trainFeaturesChunks{i}, trainLabelsChunks{i});
    if size(testFeatures, 1) > 0
        if splitTestfeatures == 1
            labels = predictIELM(net, testFeaturesChunks{i});
        else        
            labels = predictIELM(net, testFeatures);
        end        
        %sum(labels==testLabels)
        fileName = strcat(evalDstPathPrefix, '_');
        fileName = strcat(fileName, num2str(i));
        fileName = strcat(fileName, 'of');
        fileName = strcat(fileName, num2str(numOfChunks));
        fileName = strcat(fileName, '.csv');
        dlmwrite(fileName, labels);
    end
end
fileName = strcat(evalDstPathPrefix, '_');
fileName = strcat(fileName, 'of');
fileName = strcat(fileName, num2str(numOfChunks));
fileName = strcat(fileName, 'complexities.csv');
dlmwrite(fileName, nHiddenNeurons * ones(numOfChunks, 1), 'precision',10);

fileName = strcat(evalDstPathPrefix, '_');
fileName = strcat(fileName, 'of');
fileName = strcat(fileName, num2str(numOfChunks));
fileName = strcat(fileName, 'complexitiesNumParamMetric.csv');
dlmwrite(fileName, (nHiddenNeurons * size(trainFeatures, 2) + nHiddenNeurons * net.nClass) * ones(numOfChunks, 1), 'precision',10);
end

