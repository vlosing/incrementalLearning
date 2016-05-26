% UPDATERQ - Updates Rs and Q accordingly when adding or removing a
%            margin vector.  Note: Rs and Q are defined as global
%            variables.
%
% Syntax: updateRQ(beta,gamma,indc)
%         (for adding a margin vector)
%
%         updateRQ(indc)
%         (for removing a margin vector)
%
%   beta: parameter sensitivities associated with the example indc
%  gamma: margin sensitivity associated with the example indc
%   indc: example/matrix index
%
% Version 3.22e -- Comments to diehl@alumni.cmu.edu
%

function shrinkRQ(indc)
% define global variables
global Q;      % extended kernel matrix for all vectors
global Rs;     % inverse of extended kernel matrix for margin vectors
rows = size(Rs,1);
if (rows > 2)

    stripped = [1:indc-1 indc+1:size(Rs,1)];
    Rs = Rs(stripped,stripped)-Rs(stripped,indc)*Rs(indc,stripped)/Rs(indc,indc);
else
    Rs = Inf;
end;
Q(indc,:) = [];




