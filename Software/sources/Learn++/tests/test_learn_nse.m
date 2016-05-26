function test_learn_nse()
% test learn++.nse

disp('The ConceptDriftData.m file must be in the Matlab path. This');
disp('file can be found: https://github.com/gditzler/ConceptDriftData ');
addpath('../src/');

model.type = 'CART';          % base classifier
net.a = .5;                   % slope parameter to a sigmoid
net.b = 10;                   % cutoff parameter to a sigmoid
net.threshold = 0.01;         % how small is too small for error
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct


% generate the sea data set
T = 100;  % number of time stamps
N = 30;  % number of data points at each time
[data_train, labels_train,data_test,labels_test] = ConceptDriftData('noaa', T, N);

for t = 1:T
  % i wrote the code along time ago and i used at assume column vectors for
  % data and i wrote all the code for learn++ on github to assume row
  % vectors. the primary reasoning for this is that the stats toolbox in
  % matlab uses row vectors for operations like mean, cov and the
  % classifiers like CART and NB
  data_train{t} = data_train{t}';
  labels_train{t} = labels_train{t}';
  data_test{t} = data_test{t}';
  labels_test{t} = labels_test{t}';
end
net.classes = unique(labels_train{1});    
% run learn++.nse
[~,~,~,~,~,errs_nse] = learn_nse(net, data_train, labels_train, data_test, ...
   labels_test);
mean(errs_nse)
%figure;
%plot(errs_nse)

% reset the parameters of the net struct. 
model.type = 'CART';
net.a = .5;
net.b = 10;
net.threshold = 0.01; 
net.mclass = 2;
net.base_classifier = model;

% set the parameters for smote
smote_params.minority_class = 2;
smote_params.k = 3;
smote_params.N = 200;

% run learn++.cds. the difference between calling cds or nse is that the
% you pass the smote structure into the learn_nse.m function.
%[~,~,~,~,~,errs_cds] = learn_nse(net, data_train, labels_train, data_test, ...
%  labels_test, smote_params);


%figure;
%plot(errs_cds)
%mean(errs_cds)
