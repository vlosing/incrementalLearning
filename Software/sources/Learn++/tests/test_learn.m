function test_learn()
% test learn++
K = 5;

addpath('../src/');   % add the code path 
load ionosphere       % load the built in ionosphere data set
u = unique(Y);        % get the number of unique classes
labels = zeros(numel(Y), 1);

% convert the string labels to numeric labels
for n = 1:numel(Y)
  for c = 1:numel(u)
    if u{c} == Y{n}
      labels(n) = c;
      break
    end
  end
end

% shuffle the data
i = randperm(numel(Y));
data = X(i, : );
labels = labels(i);
clear Description X Y c i n u

cv = cvpartition(numel(labels),'k',K);
z = zeros(numel(labels),1);
for k = 1:K-1
  z = z + (training(cv,k)>0);
end
ts_idx = find(z == K - 1);
tr_idx = find(z ~= K - 1);

data_tr = data(tr_idx, :);
data_te = data(ts_idx, :);
labels_tr = labels(tr_idx);
labels_te = labels(ts_idx);

length(data)
length(data_tr)
length(data_te)
cv = cvpartition(numel(labels_tr),'k',K);
for k = 1:K
  data_tr_cell{k} = data_tr(training(cv,k)==0, :);
  length(data_tr_cell{k})
  labels_tr_cell{k} = labels_tr(training(cv,k)==0);
end
clear K cv data labels z tr_idx ts_idx k data_tr labels_tr 

model.type = 'CART';
net.base_classifier = model;
net.iterations = 5;
net.mclass = numel(unique(labels_te));

[net,errs] = learn(net, data_tr_cell, labels_tr_cell, data_te, labels_te);
plot(errs)