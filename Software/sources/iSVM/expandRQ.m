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

function expandRQ(beta, gamma, indc)

% define global variables
global deps;   % jitter factor in kernel matrix
global Q;      % extended kernel matrix for all vectors
global Rs;     % inverse of extended kernel matrix for margin vectors
global scale;  % kernel scale
global type;   % kernel type
global X;      % matrix of margin, error and reserve vectors stored columnwise
global y;      % column vector of class labels (-1/+1) for margin, error and reserve vectors

rows = size(Rs,1);
gamma = max(gamma, deps);
if (rows > 1)
    Rs = [Rs,zeros(rows,1);zeros(1,rows+1)] + [beta;1]*[beta',1]/gamma;
else
    Rs = [-(kernel(X(:,indc),X(:,indc),type,scale)+deps) y(indc) ; y(indc) 0];
end;
Q = [Q ; (y(indc)*y').*kernel(X(:,indc),X,type,scale)];
Q(rows+1,indc) = Q(rows+1,indc) + deps;





