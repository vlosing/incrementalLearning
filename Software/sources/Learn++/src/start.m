clear;
featuresPrefix = '/hri/storage/user/vlosing/Features/ORF/'

filenamePrefixTrain = strcat(featuresPrefix, 'Border_random_07__0')
filenamePrefixTest = filenamePrefixTrain


%filenamePrefixTrainTmp = strcat(filenamePrefix, num2str(iter-1));
%filenamePrefixTestTmp = filenamePrefixTrainTmp;
filenamePrefixTrainTmp = filenamePrefixTrain
filenamePrefixTestTmp = filenamePrefixTest

trainSamplesFileName = strcat(filenamePrefixTrainTmp, '-train.data');
testSamplesFileName = strcat(filenamePrefixTestTmp, '-test.data');
trainLabelsFileName = strcat(filenamePrefixTrainTmp, '-train.labels');
testLabelsFileName = strcat(filenamePrefixTestTmp, '-test.labels');

AllTrainSamples = dlmread(trainSamplesFileName,'', 1, 0);
AllTrainLabels = dlmread(trainLabelsFileName,';', 1, 0);
TestSamples = dlmread(testSamplesFileName,'', 1, 0);
TestLabels = dlmread(testLabelsFileName,';', 1, 0);
TestSamples = TestSamples;
classes = unique(AllTrainLabels);
clen    = length(classes);    





%numberOfChunks = 5;
%length(AllTrainSamples)
%chunkSize=length(AllTrainSamples)/numberOfChunks
%trainChunks = cell(1,numberOfChunks);
%trainLabelChunks = cell(1,numberOfChunks);

%idx=1;
%for i=1:numberOfChunks
%    trainChunks{i}=AllTrainSamples(idx:idx+chunkSize-1, :);
%    trainLabelChunks{i}=AllTrainLabels(idx:idx+chunkSize-1);
%    idx = idx + chunkSize
%end

%model.type = 'CART';
%net.base_classifier = model;
%net.iterations = 3;
%net.mclass = numel(unique(AllTrainLabels));
%[net,errs] = learn(net, trainChunks, trainLabelChunks, TestSamples, TestLabels);
%plot(errs)

K=5;
cv = cvpartition(numel(AllTrainLabels),'k',K);
for k = 1:K
  data_tr_cell{k} = AllTrainSamples(training(cv,k)==0, :);
  labels_tr_cell{k} = AllTrainLabels(training(cv,k)==0);
end
clear K cv data labels z tr_idx ts_idx k data_tr labels_tr 
model.type = 'CART';
net.base_classifier = model;
net.iterations = 3;
net.mclass = numel(unique(AllTrainLabels));

learn(net, data_tr_cell, labels_tr_cell, TestSamples, TestLabels);
plot(errs)

