function [count] = getNumberOfClassifiers()
    global net;
    count = length(net.classifiers);
end

