function [predictions,posterior] = classify_ensemble(net, data, weights)
    numExamples = size(data, 1);
    p = zeros(numExamples, numel(net.classes));
    for k = 1:length(net.classifiers)
      y = classifier_test(net.classifiers{k}, data)';
      indices = sub2ind(size(p), (1:numExamples)', y+1);
      p(indices) = p(indices) + weights(k);
    end
    [~,predictions] = max(p');
    predictions = predictions';
    predictions = net.classes(predictions);
    posterior = p./repmat(sum(p,2),1,numel(net.classes));
end

