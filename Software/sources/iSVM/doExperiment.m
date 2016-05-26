clear;


KTYPE  = 5;    % gaussian kernel
CVar = 2^10; 

%KSCALE = 0.05; 
%filenamePrefix = '../../Data/Features/ORF/COIL_randomRegular_18__';
%filenamePrefix = '../../Data/Features/ORF/OutdoorEasy_random_07__';
%filenamePrefix = '../../Data/Features/ORF/OutdoorEasy_chunksRandomEqualCountPerLabel_07__';

KSCALE = 50; 
filenamePrefix = '../../Data/Features/ORF/Border_random_07__';
%filenamePrefix = '../../Data/Features/ORF/Overlap_random_07__';

%KSCALE = 50; 
%filenamePrefix = '../../Data/Features/ORF/usps';
%KSCALE = 7500; 
%filenamePrefix = '../../Data/Features/ORF/pendigits';
%KSCALE = 40; 
%filenamePrefix = '../../Data/Features/ORF/letter';
%KSCALE = 50; 
%filenamePrefix = '../../Data/Features/ORF/dna';

iterations = 2;
results = [];
avgResults = [];

for iter=1:iterations
    filenamePrefixTmp = strcat(filenamePrefix, num2str(iter-1));
    %filenamePrefixTmp = filenamePrefix
    
    trainSamplesFileName = strcat(filenamePrefixTmp, '-train.data');
    testSamplesFileName = strcat(filenamePrefixTmp, '-test.data');
    trainLabelsFileName = strcat(filenamePrefixTmp, '-train.labels');
    testLabelsFileName = strcat(filenamePrefixTmp, '-test.labels');
    
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
    %TrainSamplesLength = [500 1000 1800];    
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
        for i=1:length(TrainLabels)
            if mod(i, 100) == 0
                disp([num2str(i)]);
            end
            currentClass = TrainLabels(i);
            for classIdx=1:clen
                if classIdx == currentClass+1
                    localLabel = 1;
                else 
                    localLabel = -1;
                end
                if (model{classIdx}.isEmpty) & (localLabel == 1)
                    model{classIdx}.isEmpty = false;
                    [alpha_coef,b_offset,derivatives,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(TrainSamples(:,i),localLabel, CVar, KTYPE, KSCALE);        
                    %save svm model (maybe not necessary after initial set...depends whether deep copy or not)
                    model{classIdx}.alpha_coef  = alpha_coef;
                    model{classIdx}.b_offset    = b_offset;
                    model{classIdx}.C           = C;
                    model{classIdx}.derivatives = derivatives;
                    model{classIdx}.indices     = ind;
                    model{classIdx}.userIndices = uind;            
                    model{classIdx}.X_mer       = X_mer;            
                    model{classIdx}.y_mer       = y_mer;            
                    model{classIdx}.Rs          = Rs;            
                    model{classIdx}.Q           = Q;                                
                elseif (~model{classIdx}.isEmpty)
                    %set svm model
                    a = model{classIdx}.alpha_coef;
                    b = model{classIdx}.b_offset;
                    C = model{classIdx}.C;
                    g = model{classIdx}.derivatives;
                    ind = model{classIdx}.indices;
                    uind = model{classIdx}.userIndices;
                    X = model{classIdx}.X_mer;
                    y = model{classIdx}.y_mer;
                    Rs = model{classIdx}.Rs;
                    Q = model{classIdx}.Q;
                    [alpha_coef,b_offset,derivatives,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(TrainSamples(:,i),localLabel, CVar);        
                    %save svm model (maybe not necessary after initial set...depends whether deep copy or not)
                    model{classIdx}.alpha_coef  = alpha_coef;
                    model{classIdx}.b_offset    = b_offset;
                    model{classIdx}.C           = C;
                    model{classIdx}.derivatives = derivatives;
                    model{classIdx}.indices     = ind;
                    model{classIdx}.userIndices = uind;            
                    model{classIdx}.X_mer       = X_mer;            
                    model{classIdx}.y_mer       = y_mer;            
                    model{classIdx}.Rs          = Rs;            
                    model{classIdx}.Q           = Q;                                
                end
            end  
        end
        for classIdx=1:clen
            if ~model{classIdx}.isEmpty
                allMargin = allMargin + length(model{classIdx}.indices{MARGIN});
                allError = allError + length(model{classIdx}.indices{ERROR});
                [margin_test(:,classIdx)]  = svmeval(TestSamples,model{classIdx}.alpha_coef,model{classIdx}.b_offset,model{classIdx}.indices,model{classIdx}.X_mer,model{classIdx}.y_mer,KTYPE,KSCALE);                
                [margin_train(:,classIdx)] = svmeval(TrainSamples,model{classIdx}.alpha_coef,model{classIdx}.b_offset,model{classIdx}.indices,model{classIdx}.X_mer,model{classIdx}.y_mer,KTYPE,KSCALE);                
            else
                disp(['no train samples for class ', num2str(tt)]);
                [margin_test(:,tt)]  = ones(length(TestSamples),1)*-9999;
            end
        end
        [dummy,l_est_train] = max(margin_train, [], 2);
        [dummy,l_est_test]  = max(margin_test, [], 2);
        l_est_train = l_est_train - 1;
        l_est_test = l_est_test - 1;
        
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
