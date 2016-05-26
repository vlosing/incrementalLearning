clc;
clear;
close all;

addpath('../src/');

n_observations = 1000;
n_features = 10;
window = 100;
data = randn(n_observations,n_features);
labels = randi(2,1,n_observations)';
[x1,x2,y1,y2] = test_then_train(data, labels, window);