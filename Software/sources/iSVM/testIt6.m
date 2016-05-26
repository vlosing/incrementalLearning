clear;
featuresPrefix = '/hri/storage/user/vlosing/Features/ORF'


KTYPE  = 5;    % gaussian kernel
CVar = 2^10; 

%KSCALE = 0.05; 
%filenamePrefixTrain = '../../Data/Features/ORF/COIL_randomRegular_18__';
%filenamePrefixTrain = '../../Data/Features/ORF/OutdoorEasy_random_07__';
%filenamePrefixTrain = '../../Data/Features/ORF/OutdoorEasy_chunksRandomEqualCountPerLabel_07__';

KSCALE = 50; 
filenamePrefixTrain = strcat(featuresPrefix, '/Border_random_kFold_0');

%filenamePrefixTrain = strcat(featuresPrefix, '/Overlap_random_07__');
filenamePrefixTest = filenamePrefixTrain;

%KSCALE = 50; 
%filenamePrefixTrain = '../../Data/Features/ORF/usps';
%KSCALE = 7500; 
%filenamePrefixTrain = '../../Data/Features/ORF/pendigits';
%KSCALE = 40; 
%filenamePrefixTrain = '../../Data/Features/ORF/letter';
%KSCALE = 50; 
%filenamePrefixTrain = '../../Data/Features/ORF/dna';
%filenamePrefixTest = filenamePrefixTrain

%KSCALE = 1500; 
%filenamePrefixTrain = '/hri/storage/user/vlosing/Features/ORF/2X_EU_001-086_64000_100_100_1_random_1_0_0';
%filenamePrefixTest = '/hri/storage/user/vlosing/Features/ORF/CutInTest_original_1_0_0';


iterations = 1;
results = [];
avgResults = [];

for iter=1:iterations
    %filenamePrefixTrainTmp = strcat(filenamePrefixTrain, num2str(iter-1));
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
    TestSamples = TestSamples';
    classes = unique(AllTrainLabels);
    clen    = length(classes);    
    
    disp([num2str(clen), ' classes'])    
    disp([num2str(length(TestLabels)), ' test samples'])    
    


    TrainSamplesLength = [length(AllTrainSamples)];
    %TrainSamplesLength = [1000];    
    for sampleCount=1:length(TrainSamplesLength)
        MaxTrainLength = TrainSamplesLength(sampleCount);
        TrainSamples = AllTrainSamples(1:MaxTrainLength,:);
        TrainLabels = AllTrainLabels(1:MaxTrainLength,:);
        TrainLabelsLocal = TrainLabels; 
        model   = cell(1,clen);
        for i=1:clen
            model{i}.isEmpty = true;
        end
        
        margin_test  = zeros(length(TestLabels),clen);
        margin_train = zeros(length(TrainLabels),clen);
        % one versus rest
        TrainSamples = TrainSamples';
        disp([num2str(length(TrainSamples)), ' train samples'])    

        allMargin = 0;
        allError = 0;
        
        % flags for example state
        MARGIN    = 1;
        ERROR     = 2;
        RESERVE   = 3;
        UNLEARNED = 4;

        global a;                     % alpha coefficients
        global b;                     % bias
        global C;                     % regularization parameters 
        global deps;                  % jitter factor in kernel matrix
        global g;                     % partial derivatives of cost function w.r.t. alpha coefficients
        global ind;                   % cell array containing indices of margin, error, reserve and unlearned vectors
        global kernel_evals;          % kernel evaluations
        global max_reserve_vectors;   % maximum number of reserve vectors stored
        global perturbations;         % number of perturbations
        global Q;                     % extended kernel matrix for all vectors
        global Rs;                    % inverse of extended kernel matrix for margin vectors   
        global scale;                 % kernel scale
        global type;                  % kernel type
        global uind;                  % user-defined example indices
        global X;                     % matrix of margin, error, reserve and unlearned vectors stored columnwise
        global y;                     % column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors    
        max_reserve_vectors = 200;
        max_supportVectors = 1000;
        for i=1:length(TrainLabels)
            if mod(i, 100) == 0
                disp([num2str(i)]);
            end
            svCount = 0;
            currentClass = TrainLabels(i);
            for classIdx=1:clen
                if classIdx == currentClass+1
                    localLabel = 1;
                else 
                    localLabel = -1;
                end
                if (model{classIdx}.isEmpty) & (localLabel == 1)
                    model{classIdx}.isEmpty = false;
                    model{classIdx}.offset = i-1;
                    svmtrain(TrainSamples(:,i),localLabel, CVar, KTYPE, KSCALE);        
                    model{classIdx} = updateModel(model{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
                    svCount = svCount + length(model{classIdx}.ind{MARGIN}) + length(model{classIdx}.ind{ERROR});
                elseif (~model{classIdx}.isEmpty)
                    modelToParams(model{classIdx})
                    svmtrain(TrainSamples(:,i),localLabel, CVar);        
                    %save svm model (maybe not necessary after initial set...depends whether deep copy or not)
                    model{classIdx} = updateModel(model{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
                    svCount = svCount + length(model{classIdx}.ind{MARGIN}) + length(model{classIdx}.ind{ERROR});
                end
                %doesnt work properly
                %{
                while svCount > max_supportVectors
                    minAlpha = 999999;
                    minClassIdx = -1;
                    minSampleIdx = -1;
                    absSampleIdx = -1;
                    for classIdx=1:clen
                        if ~model{classIdx}.isEmpty
                            nonZeroIndices = [model{classIdx}.ind{MARGIN} model{classIdx}.ind{ERROR}];
                            if length(nonZeroIndices) > 0
                                model{classIdx}.a(nonZeroIndices);
                                [minValue, index] = min(model{classIdx}.a(nonZeroIndices));
                                if minValue < minAlpha & length(model{classIdx}.ind{MARGIN})>2 %avoid removing all margin vectors of one SVM
                                    minAlpha = minValue;
                                    minSampleIdx = nonZeroIndices(index);
                                    minClassIdx = classIdx;
                                    absSampleIdx = minSampleIdx + model{classIdx}.offset;
                                end
                            end
                        end
                    end

                    if minSampleIdx > 0
                        for classIdx=1:clen
                            if ~model{classIdx}.isEmpty
                                %set...depends whether deep copy or not)
                                modelToParams(model{classIdx})
                                %disp(['before ',
                                %num2str(length(model{minClassIdx}.ind{MARGIN}))]);
                                
                                %unlearn(minSampleIdx);
                                if (absSampleIdx-model{classIdx}.offset) > 0
                                    unlearn(absSampleIdx-model{classIdx}.offset);
                                    %save svm model (maybe not necessary after initial
                                    model{classIdx} = updateModel(model{classIdx}, a, b, C, deps, g, ind, kernel_evals, perturbations, Q, Rs, scale, type, uind, X, y);
                                    
                                    %disp(['after ', num2str(length(model{minClassIdx}.ind{MARGIN}))]);
                                end
                            end
                        end
                        svCount = 0;                        
                        for classIdx=1:clen
                            if ~model{classIdx}.isEmpty
                                svCount = svCount + length(model{classIdx}.ind{MARGIN}) + length(model{classIdx}.ind{ERROR});
                            end
                        end
                    else
                        break;
                    end
                end
                %}

            end  
        end
        for classIdx=1:clen
            if ~model{classIdx}.isEmpty
                disp(['class ', num2str(classIdx), ' margin ',  num2str(length(model{classIdx}.ind{MARGIN})), ' error ', num2str(length(model{classIdx}.ind{ERROR}))]);
            end                    
        end        
        for classIdx=1:clen
            if ~model{classIdx}.isEmpty
                allMargin = allMargin + length(model{classIdx}.ind{MARGIN});
                allError = allError + length(model{classIdx}.ind{ERROR});
                model{classIdx}
                [margin_test(:,classIdx)]  = svmeval(TestSamples,model{classIdx}.a,model{classIdx}.b,model{classIdx}.ind,model{classIdx}.X,model{classIdx}.y,type,scale);
                [margin_train(:,classIdx)] = svmeval(TrainSamples,model{classIdx}.a,model{classIdx}.b,model{classIdx}.ind,model{classIdx}.X,model{classIdx}.y,type,scale);
            else
                disp(['no train samples for class ', num2str(classIdx)]);
                [margin_test(:,classIdx)]  = ones(length(TestSamples),1)*-9999;
            end
        end
        [dummy,l_est_train] = max(margin_train, [], 2);
        [dummy,l_est_test]  = max(margin_test, [], 2);
        l_est_train = l_est_train - 1;
        l_est_test = l_est_test - 1;
        %l_est_test
        
        trainError       = sum(l_est_train~=TrainLabels)*100/length(TrainLabels);
        testError        = sum(l_est_test~=TestLabels)*100/length(TestLabels);
        disp(['trainError ', num2str(trainError), ' testError ', num2str(testError)]);
        disp(['allMargin ', num2str(allMargin),' allError ', num2str(allError)]);
        results = [results; iter MaxTrainLength testError allMargin + allError];
    end
end
results;
for sampleCount=1:length(TrainSamplesLength)
    indices = find(results(:,2) == TrainSamplesLength(sampleCount));
    avgResults = [avgResults; TrainSamplesLength(sampleCount) sum(results(indices,3:4),1)/iterations];
end
disp(['avgResults ', num2str(avgResults)]);
