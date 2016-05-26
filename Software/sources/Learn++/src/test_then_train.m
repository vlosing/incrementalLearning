function [data_train,data_test,labels_train,labels_test] = test_then_train(data, labels, win_size)
%   [data_train,data_test,labels_train,labels_test] = ...
%       test_then_train(data, labels, win_size);
%    
%     @data - data in n_observations by n_features matrix
%     @labels - labels in n_observations by 1 vector
%     @win_size - batch size
%     @data_train - cell array of training data
%     @data_test - cell array of test data
%     @labels_train - cell array of training labels
%     @labels_test - cell array of test labels
%     
%     
%   Partition a data set in fixed length windows for training and 
%   testing. 
%  
%   @Author: Gregory Ditzler (gregory.ditzler@gmail.com) 
%      

%     test_then_train.m
%     Copyright (C) 2013 Gregory Ditzler
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
n_observations = length(labels);
n = 0;
kill_loop = false;
data_train = {};
data_test = {};
labels_train = {};
labels_test = {};

while true
  te_idx = (n+1)*win_size+1:(n+2)*win_size;
  tr_idx = n*win_size+1:(n+1)*win_size;
  n = n + 1;
  if max(te_idx) > n_observations
    te_idx = te_idx(te_idx <= n_observations);
    kill_loop = true;
  end 
  
  data_train{n} = data(tr_idx, :);
  data_test{n} = data(te_idx, :);
  labels_train{n} = labels(tr_idx);
  labels_test{n} = labels(te_idx);

  if kill_loop == true
    break;
  end
end