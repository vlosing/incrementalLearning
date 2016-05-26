function model = classifier_train(model, data, labels)
%   model = CLASSIFIER_TRAIN(model, data)
%     @model - structure. you must set the type field. Only CART is 
%       implemented at the moment. 
%       (manditory fields)
%         > .type - 'CART'
%         (optional fields)
%           > .prune - see CLASSREGTREE.m
%           > .minleaf - see CLASSREGTREE.m
%           > .mergeleaves - see CLASSREGTREE.m
%           > .surrogate - see CLASSREGTREE.m
%     @data - matrix of data. n_observations by n_features
%     @labels - vector of labels made by the classifier
%     @model - update model structure with a trained classifier
%     
%   Train a classifier on labeled data.
% 
%   @Author: Gregory Ditzler (gregory.ditzler@gmail.com)
% 
%   See also
%   classifier_test.m

%     classifier_train.m
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


% You can easily modify this file to include new baseclassifiers. This file
% must have lines added for creating the classifier that you are interested
% in. You also need to modify CLASSIFIER_TEST.M to implement the evaluation
% of the classifier on a test set. 

[n_observations, n_features] = size(data);
if length(labels) ~= n_observations
  error(['CLASSIFIER_TRAIN.M :: Data obervations not equal to the number',...
    ' of labels'])
end

switch model.type
  case 'CART'
    % the CART classifier is the base model
    model.n_features = n_features;
    model.method = 'classification';
    
    if isfield(model, 'prune') == 0
      % compute the full tree and the optimal sequence of pruned 
      % subtrees, or 'off' for the full tree without pruning.
      model.prune = 'on';
    end
    if isfield(model, 'minleaf') == 0
      %  minimal number of observations per tree leaf
      model.minleaf = 1;
    end
    if isfield(model, 'mergeleaves') == 0
      % rge leaves that originate from the same parent node and give the 
      % sum of risk values greater or equal to the risk associated with the 
      % parent node.
      model.mergeleaves = 'on';
    end
    if isfield(model, 'surrogate') == 0
      % recall that surrogate tree are for missing data. the classregtree
      % impementation is generally much slower when this is set to on. and
      % it is on bydefault. lets turn it off!
      model.surrogate = 'off';
    end
    
    model.classifier = classregtree(data, labels, ...
      'method', model.method,...
      'prune', model.prune,...
      'minleaf', model.minleaf,...
      'mergeleaves', model.mergeleaves,...
      'surrogate', model.surrogate);
    
end