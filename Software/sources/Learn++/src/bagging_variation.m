function sub_ensemble = bagging_variation(data, labels, n_classifiers, minority_class, base_classifier)
%   sub_ensemble = bagging_variation(data, labels, ...
%     n_classifiers, minority_class, base_classifier)
% 
%     @data - data matrix in n_observations by number of features 
%       matrix
%     @labels - labels in an n_observations by one vector
%     @n_classifiers - number of classifiers to generate in the 
%       subensemble
%     @minority_class - integer specifying which class in the 
%       prediction problem is the minority
%     @base_classifier - base classification algorithm used in 
%       CLASSIFIER_TRAIN.m
%     @sub_ensemble - cell array containing the sub ensemble of 
%       classifiers.
%     
%   This function implements the bagging variation algorithm used in 
%   Learn++.NIE. 
% 
%   Cite: 
%   1) G. Ditzler and R. Polikar, "Incremental learning of concept drift 
%      from streaming imbalanced data," in IEEE Transactions on Knowledge 
%      & Data Engineering, 2012, accepted.
% 
%   @Author: Gregory Ditzler (gregory.ditzler@gmail.com) 


%     bagging_variation.m
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

  
negative_indices = find(labels ~= minority_class);
positive_indices = find(labels == minority_class);
sub_ensemble = cell(n_classifiers, 1);

parfor k = 1:n_classifiers
  index = negative_indices(randi(numel(negative_indices), 1, ...
    floor(numel(labels)/n_classifiers)));
  data_k = [data(index, :); data(positive_indices, :)];
  label_k = [labels(index); labels(positive_indices)];
  sub_ensemble{k} = classifier_train(base_classifier, data_k, label_k);
end