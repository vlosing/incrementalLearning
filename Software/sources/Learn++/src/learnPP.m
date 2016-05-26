function learnPP(data_train, labels_train)
%   [net,errs] = learn(net, data_train, labels_train, ...
%     data_test, labels_test)
%
%     @net - initialized structure. you must initialize
%       net.iterations
%       net.base_classifier - you should set this to be model.type
%         which is submitted to CLASSIFIER_TRAIN.m
%       net.mclass - number of classes
%     @data_train - training data in a cell array. each entry should
%         have a n_oberservation by n_feature matrix
%     @labels_train - cell array of class labels
%     @data_test - test data in a matrix. the size of the matrix should
%         be n_oberservation by n_feature matrix
%     @labels_test -labels to the test data
%     @errs - error of the Learn++ on the testing data set. error is
%       measured at each addition of a new classifier.
%
%   Implementation of Learn++.
%
%   Cite:
%   1) R. Polikar, L. Udpa, S. Udpa, and V. Honavar, "Learn++: An
%      incremental learning algorithm for supervised neural networks,"
%      IEEE Transactions on System, Man and Cybernetics (C), Special
%      Issue on Knowledge Management, vol. 31, no. 4, pp. 497-508, 2001.
%
%   See also
%   CLASSIFIER_TRAIN.m CLASSIFIER_TEST.m

%     learn.m
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
global net;
c_count = length(net.classifiers);
data_train_k = data_train;
size(data_train)
labels_train_k = labels_train;
D = ones(numel(labels_train_k), 1)/numel(labels_train_k);

% original paper says to modify D if prior knowledge is available. we can
% modify the distribution weights if we already have a classifier
% ensemble.
if net.initialized==1
    predictions = classify_ensemble(net, data_train_k, log(1./net.beta(1:length(net.classifiers))));   % predict on the training data
    if sum(predictions ~= labels_train_k) == 0
        disp(['already perfect accuracy']);
        return;
    end
    epsilon_kt = sum(D(predictions ~= labels_train_k)); % error on D
    beta_kt = epsilon_kt/(1-epsilon_kt);                % normalized error on D
    D(predictions == labels_train_k) = beta_kt * D(predictions == labels_train_k);
end

for t = 1:net.iterations
    % update the classifier count
    c_count = c_count + 1;
    
    
    % step 1 - make sure we are working with a probability distribution.
    D = D / sum(D);
    % step 2 - grab a random sample of data indices with replacement from
    % the probability distribution D
    
    index = randsample(1:numel(D), numel(D), true, D);
    
    % step 3 - generate a new classifier on the data sampled from D.
    net.classifiers{c_count} = classifier_train(...
        net.base_classifier, ...
        data_train_k(index, :), ...
        labels_train_k(index));
    
    % step 4 - test the latest classifier on ALL of the data not just the
    % data sampled from D, and compute the error according to the
    % probability distribution. then compute beta
    y = classifier_test(net.classifiers{c_count}, data_train_k);
        
    epsilon_kt = sum(D(y ~= labels_train_k));
    net.beta = [net.beta;
        epsilon_kt/(1-epsilon_kt)];
    % step 5 - get the ensemble decision computed with c_count classifiers
    % in the ensemble. compute the error on the probability distribution on
    % the composite hypothesis.
    predictions = classify_ensemble(net, data_train_k, log(1./net.beta(1:length(net.classifiers))));
   
    if sum(predictions ~= labels_train_k) > 0
        E_kt = sum(D(predictions ~= labels_train_k));
        
        if E_kt > 0.5
            % rather than remove remove existing classifier; null the result out
            % by forcing the loss to be equal to 1/2 which is the worst possible
            % loss. feel free to modify the code to go back an iteration.
            E_kt = 0.5;
            %disp(['too bad']);
        end
        
        % step 6 - compute the normalized error of the compsite hypothesis and
        % update the weights over the training instances in the kth batch.
        Bkt = E_kt / (1 - E_kt);
        
        D(predictions == labels_train_k) = Bkt * D(predictions == labels_train_k);
        D = D / sum(D);
    else
        break;
    end
end
net.initialized = 1;
end





