function [T,D] = Wdist_LP(M, a, b)
% Find the optimal solutions to the optimal tranport problems using linear
% programming.
% Input:        M -- the n x m cost matrix of X (d x n matrix) and Y (d x m matrix)
%               lambda -- the regularization parameter > 0
%               a -- the weight vector of X
%               b -- the weight vector of Y
    options = optimoptions('linprog','Display','none');
    [n, m] = size(M);
    f = reshape(M, [], 1);
    Aeq = zeros(n + m + 1, n * m);
    for k = 1 : n
        Aeq(k, k:n:end) = 1; % sum of each row is ak
    end
    for l = 1 : m
        Aeq(n + l, (l-1)*n+1:l * n) = 1; % sum of each column is bl
    end
    Aeq(n + m + 1,:) = ones(1,n*m);  % total sum is 1
    beq = [a;b;1];
    lb = zeros(n*m,1);
    ub = ones(n*m, 1);
    [x, D] = linprog(f,[],[],Aeq,beq,lb,ub,options);
    T = reshape(x, n, m);
end