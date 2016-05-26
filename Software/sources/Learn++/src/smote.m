function SMOTEd = smote(data, k, N)
%   data = SMOTE(data, k, N)
%     @data - matrix of data. n_observations by n_features
%     @k - nearest neighbors
%     @model - smote precentage
%     
%   Run SMOTE on data.
% 
%   @Author: Gregory Ditzler (gregory.ditzler@gmail.com)

%     smote.m
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


T = size(data, 1);
SMOTEd = [];

% If N is less than 100%, randomize the minority class samples as only a 
% random percent of them will be SMOTEd
if N < 100,
  T = round((N/100)*T);
  N = 100;
end

for i = 1:T,
  nnarray = [];
  synthetic = [];

  % determine the euclidean distance between the current minority sample
  % and reset of the other minority samples. then sort them in ascending
  % order
  for j = 1:T,
    if i ~= j 
      euclid_dist(j,:) = data(i,:) - data(j,:);
    else
      % ignore the sample we are currently at from further calculations. 
      euclid_dist(j,:) = inf * ones(1, size(data,2));
    end
  end
  euclid_dist = sqrt(sum(euclid_dist.^2,2));
  euclid_dist2 = sort(euclid_dist,'ascend');

  % if we a really dealing with an imbalanced data set we may not have
  % enough samples to reduce to k nearest neighbors; instead grab all of
  % them 
  if length(euclid_dist2)<=k,
    knn = euclid_dist2;
    k = length(euclid_dist2);
  else
    knn = euclid_dist2(1:k);
  end

  % determine the k-nearest neighbors to the minority sample that we are
  % interested in.
  for j = 1:length(euclid_dist),
    if sum(euclid_dist(j)==knn)
      % the current distance in euclid_dist is a nearest neigbor of
      % the minority sample. so nnarray will have the indices of the
      % nearest neighbors in the minority instance array.
      nnarray(end+1) = j;
    end
  end

  % generate the synthetic samples
  newindex = 1; % keeps a count of number of synthetic samples generated
  N1 = round(N/100); % DO NOT OVERWRITE N!!!!

  while N1~=0,
      % Choose a random number between 1 and k, call it nn. This step 
    % chooses one of the k nearest neighbors of i
    nn = round((k-1)*rand+1); % perform a linear conversion to scale the
                              % nn paramter between 1 and k
    gap = rand;
    dif = data(nnarray(nn), :) - data(i, :);
    synthetic(newindex,:) = data(i, :) + gap*dif;

    newindex = newindex+1;
    N1 = N1-1;
  end
  SMOTEd = [SMOTEd; synthetic];
  clear euclid_dist euclid_dist2 N1 nnarray synthetic nnarray
end