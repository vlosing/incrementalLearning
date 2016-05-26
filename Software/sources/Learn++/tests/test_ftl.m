clc;
clear;
close all;


disp('The ConceptDriftData.m file must be in the Matlab path. This');
disp('file can be found: https://github.com/gditzler/ConceptDriftData ');
addpath('../src/');

model.type = 'CART';          % base classifier
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct
net.n_classifiers = 10;

% generate the sea data set
T = 200;  % number of time stamps
N = 100;  % number of data points at each time
[data_train, labels_train,data_test,labels_test] = ConceptDriftData('sea', T, N);
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

% run learn++.nse
[net,f_measure,g_mean,precision,recall,err] = follow_the_leader(net, data_train, labels_train, data_test, ...
   labels_test);

 plot(err)