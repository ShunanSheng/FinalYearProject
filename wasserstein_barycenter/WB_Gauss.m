function [WB, cost] = WB_Gauss(input_cell, lambda, tol)
% Compute the Wasserstein-2 barycenter of Gaussian measures by a
% fixed-point algorithm
% Inputs: 
%       input_cell: cell array containing structs with fields mu and Sig
%       where mu stands for the mean vector and Sig stands for the
%       covariance matrix
%       lambda: the weights defining the Wasserstein barycenter
%       tol: the tolerance threshold; when the change in the cost is less
%       than tol, the algorithm terminates (default is 1e-6)
% Outputs: 
%       WB: struct with fields mu and Sig
%       cost: the minimized cost value

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-6;
end

N = length(input_cell);

assert(length(lambda) == N, 'measures and weights mis-match');

d = length(input_cell{1}.mu);

WB = struct;
WB.mu = zeros(d, 1);

% the mean vector of the Wasserstein barycenter is simply the barycenter of
% the mean vectors of the input measures
for i = 1:N
    WB.mu = WB.mu + lambda(i) * input_cell{i}.mu;
end

% compute the cost related to mean vectors and the trace of the covariance
% matrices
cost = 0;
for i = 1:N
    cost = cost + lambda(i) * (sum((input_cell{i}.mu - WB.mu) .^ 2) ...
        + trace(input_cell{i}.Sig)); 
end

S = eye(d);
cost_var = inf;

while true
    S_sqrt = sqrtm(S);
    T = zeros(d, d);
    
    for i = 1:N
        T = T + lambda(i) * sqrtm(S_sqrt * input_cell{i}.Sig * S_sqrt);
    end
    
    cost_var_new = trace(S) - 2 * trace(T);
    
    if abs(cost_var_new - cost_var) < tol
        break;
    end
    cost_var = cost_var_new;
    
    M = S_sqrt \ T;
    
    S = M * M';
end

WB.Sig = S;
cost = cost + cost_var;

end

