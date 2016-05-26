function predictions = classifier_test(model, data)
%   predictions = CLASSIFIER_TEST(model, data)
%     @model - you get this structure from CLASSIFIER_TRAIN.m
%     @data - matrix of data. n_observations by n_features
%     @predictions - vector of predictions made by the classifier
%       trained in the CLASSIFIER_TRAIN.m function
% 
%   Test a classifier on data.
%
%   @Author: Gregory Ditzler (gregory.ditzler@gmail.com)
%
%   See also
%   classifier_train.m

%     classifier_test.m
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
% in. You also need to modify CLASSIFIER_TRAIN.M to implement the evaluation
% of the classifier on a test set. 

predictions_raw = model.classifier(data);
predictions = reshape(str2num(strjoin(predictions_raw)), size(predictions_raw))';
