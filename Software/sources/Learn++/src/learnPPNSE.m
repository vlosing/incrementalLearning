function [net,f_measure,g_mean,precision,recall,err] = learnPPNSE(net, data_train, labels_train)
%    [net,f_measure,g_mean,precision,recall,err] = learn_nse(net, ...
%        data_train, labels_train, ...
%        data_test, labels_test, ...
%        smote_params)
%
%     @net - initialized structure. you must initialize
%       net.a - sigmoid slope (try 0.5)
%       net.b - sigmoid cutoff (try 10)
%       net.threshold - small error threshold (try 0.01)
%       net.mclass - number of classes
%       net.base_classifier - you should set this to be model.type
%         which is submitted to CLASSIFIER_TRAIN.m
%     @data_train - cell array of training data. each entry should
%       have a n_oberservation by n_feature matrix
%     @labels_train - cell array of class labels
%     @data_test - cell array of training data. each entry should
%       have a n_oberservation by n_feature matrix
%     @labels_test - cell array of class labels
%     @smote_params - optional structure for implementing learn++.cds
%         smote_params.minority_class - minority class (scalar)
%         smote_params.k - see SMOTE.m
%         smote_params.N - see SMOTE.m
%
%   Implementation of Learn++.NSE and Learn++.CDS. If @smote_params
%   is specified then the implementation is Learn++.CDS
%
%   Cite:
%   1) Elwell R. and Polikar R., "Incremental Learning of Concept Drift
%      in Nonstationary Environments" IEEE Transactions on Neural Networks,
%      vol. 22, no. 10, pp. 1517-1531
%   2) G. Ditzler and R. Polikar, "Incremental learning of concept drift
%      from streaming imbalanced data," in IEEE Transactions on Knowledge
%      & Data Engineering, 2012, accepted.
%
%   @Author: Gregory Ditzler (gregory.ditzler@gmail.com)
%
%   See also
%   SMOTE.m CLASSIFIER_TRAIN.m CLASSIFIER_TEST.m



%     learn_nse.m
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


% get the training data for the 't'th round
data_train_t = data_train;

labels_train_t = labels_train;

mt = size(data_train_t,1); % numnber of training examples
Dt = ones(mt,1)/mt;         % initialize instance weight distribution

if net.initialized==1
    % STEP 1: Compute error of the existing ensemble on new data
    predictions = classify_ensemble(net, data_train_t, net.w(end,:));
    Et = sum((predictions~=labels_train_t)/mt);
    Bt = Et/(1-Et);           % this is suggested in Metin's IEEE Paper
    if Bt==0, Bt = 1/mt; end; % clip
    
    % update and normalize the instance weights
    Dt(predictions==labels_train_t) = Dt(predictions==labels_train_t) * Bt;
    Dt = Dt/sum(Dt);
end

% STEP 3: New classifier
net.classifiers{end + 1} = classifier_train(net.base_classifier, data_train_t, labels_train_t);

% STEP 4: Evaluate all existing classifiers on new data
t = size(net.classifiers,2);
y = decision_ensemble(net, data_train_t, labels_train_t, t);

for k = 1:net.t
    epsilon_tk  = sum(Dt(y(:, k) ~= labels_train_t));
    if (k<net.t)&&(epsilon_tk>0.5)
        epsilon_tk = 0.5;
    elseif (k==net.t)&&(epsilon_tk>0.5)
        % try generate a new classifier
        net.classifiers{k} = classifier_train(net.base_classifier, data_train_t, labels_train_t);
        epsilon_tk  = sum(Dt(y(:, k) ~= labels_train_t));
        epsilon_tk(epsilon_tk > 0.5) = 0.5;   % we tried; clip the loss
    end
    net.beta(net.t,k) = epsilon_tk / (1-epsilon_tk);
end

% compute the classifier weights
if net.t==1,
    if net.beta(net.t,net.t)<net.threshold,
        net.beta(net.t,net.t) = net.threshold;
    end
    net.w(net.t,net.t) = log(1/net.beta(net.t,net.t));
else
    for k = 1:net.t,
        b = t - k - net.b;
        omega = 1:(net.t - k + 1);
        omega = 1./(1+exp(-net.a*(omega-b)));
        omega = (omega/sum(omega))';
        beta_hat = sum(omega.*(net.beta(k:net.t,k)));
        if beta_hat < net.threshold,
            beta_hat = net.threshold;
        end
        net.w(net.t,k) = log(1/beta_hat);
    end
end

% STEP 7: classifier voting weights
net.classifierweigths{end+1} = net.w(end,:);
net.t = net.t + 1;
net.initialized = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUXILARY FUNCTIONS
function y = decision_ensemble(net, data, labels, n_experts)
y = zeros(numel(labels), n_experts);
for k = 1:n_experts
    y(:, k) = classifier_test(net.classifiers{k}, data);
end
end
