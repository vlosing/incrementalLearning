function [nodeNumber] = getNumberOfNodes()
    global net;
    nodeNumber = 0;
    for k = 1:length(net.classifiers)
        nodeNumber = nodeNumber + numnodes(net.classifiers{k}.classifier);
    end
end

