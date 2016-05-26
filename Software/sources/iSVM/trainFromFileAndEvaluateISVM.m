function trainFromFileAndEvaluateISVM(trainFeaturesFileName, trainLabelsFileName,testFeaturesFileName, evaluationStepSize, evalDstPathPrefix, splitTestfeatures, kernel, sigma, C, maxReserveVectors)
rng('shuffle');
C = double(C);
sigma = double(sigma);
if strcmp(kernel, 'linear') == 1
    kernelType = 1;
elseif strcmp(kernel, 'RBF') == 1
    kernelType = 5;
else
    disp(['unknown kernel ', kernel]);
end
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
numOfChunks = size(trainFeaturesChunks, 2);
[numOfClasses, models] = initSVM(trainLabels);
complexities = zeros(numOfChunks,1);
complexityNumParameterMetric = zeros(numOfChunks,1);
for i=1:numOfChunks
    disp(['chunk', num2str(i), '/',num2str(numOfChunks)]);
    trainFeaturesChunks{i} = trainFeaturesChunks{i}';
    if i > 1
        labels = predictISVM(trainFeaturesChunks{i}, models);
        fileName = strcat(evalDstPathPrefix, '_');
        fileName = strcat(fileName, num2str(i));
        fileName = strcat(fileName, 'of');
        fileName = strcat(fileName, num2str(numOfChunks));        
        fileName = strcat(fileName, 'predictedTrainLabels.csv');
        dlmwrite(fileName, labels);
    end
    [models, numMarginVectors, numErrorVectors] = fitiSVM(trainFeaturesChunks{i}, trainLabelsChunks{i}, models, numOfClasses, kernelType, C, sigma, maxReserveVectors);
    if ~isempty(testFeaturesFileName)
        if splitTestfeatures == 1
            labels = predictISVM(testFeaturesChunks{i}', models);
        else
            labels = predictISVM(testFeatures', models);
        end        
        fileName = strcat(evalDstPathPrefix, '_');
        fileName = strcat(fileName, num2str(i));
        fileName = strcat(fileName, 'of');
        fileName = strcat(fileName, num2str(numOfChunks));
        fileName = strcat(fileName, '.csv');
        dlmwrite(fileName, labels);
    end
    complexities(i) = numMarginVectors + numErrorVectors;
    complexityNumParameterMetric(i) = complexities(i) * (size(trainFeatures, 2) + 1) + 1;
%numMarginVectors
%numErrorVectors
end
%numMarginVectors
%numErrorVectors
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

