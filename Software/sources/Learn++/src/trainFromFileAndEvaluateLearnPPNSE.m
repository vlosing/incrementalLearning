function trainFromFileAndEvaluateLearnPPNSE(trainFeaturesFileName, trainLabelsFileName,testFeaturesFileName, evaluationStepSize, evalDstPathPrefix, splitTestfeatures)
rng('shuffle');
trainFeatures = dlmread(trainFeaturesFileName,'', 1, 0);
trainLabels = dlmread(trainLabelsFileName,'', 1, 0);
if ~isempty(testFeaturesFileName)
    testFeatures = dlmread(testFeaturesFileName,'', 1, 0);
    if splitTestfeatures == 1
        testFeaturesChunks = splitArrayBySize(testFeatures, evaluationStepSize);
    end    
end
trainFeaturesChunks = splitArrayBySize(trainFeatures, evaluationStepSize);
trainLabelsChunks = splitArrayBySize(trainLabels, evaluationStepSize);
net = initLearnPPNSE(trainLabels, .5, 10,  0.01);
numOfChunks = size(trainFeaturesChunks, 2);
complexities = zeros(numOfChunks,1);
complexityNumParameterMetric = zeros(numOfChunks,1);
for i=1:numOfChunks
    disp(['chunk', num2str(i), '/', num2str(numOfChunks)]);
    if i > 1
        [labels, posterior] = classify_ensemble(net, trainFeaturesChunks{i}, net.w(end,:));
        fileName = strcat(evalDstPathPrefix, '_');
        fileName = strcat(fileName, num2str(i));
        fileName = strcat(fileName, 'of');
        fileName = strcat(fileName, num2str(numOfChunks));        
        fileName = strcat(fileName, 'predictedTrainLabels.csv');
        dlmwrite(fileName, labels);
    end
    net = learnPPNSE(net, trainFeaturesChunks{i}, trainLabelsChunks{i});
    numLeaves = getNumberOfLeaves(net);
    numOfNodes = getNumberOfNodes(net);
    complexities(i) = numOfNodes;
    complexityNumParameterMetric(i) = numOfNodes + (numOfNodes - numLeaves) * 2 + numLeaves + i;
    if ~isempty(testFeaturesFileName)
        if splitTestfeatures == 1
            [labels, posterior] = classify_ensemble(net, testFeaturesChunks{i}, net.w(end,:));
        else        
            [labels, posterior] = classify_ensemble(net, testFeatures, net.w(end,:));
        end
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
dlmwrite(fileName, complexities, 'precision',10);

fileName = strcat(evalDstPathPrefix, '_');
fileName = strcat(fileName, 'of');
fileName = strcat(fileName, num2str(numOfChunks));        
fileName = strcat(fileName, 'complexitiesNumParamMetric.csv');
dlmwrite(fileName, complexityNumParameterMetric, 'precision',10);
end

