function [leaveNumber] = getNumberOfLeaves()
    global net;
    leaveNumber = 0;
    for k = 1:length(net.classifiers)
        leaveNumber = leaveNumber + floor(numnodes(net.classifiers{k}.classifier)/2) + 1;
    end
end

