function a_hat = WB_fixed_support(X, Y_cell, b_cell, t0, tol, M_cell, Max_Iter)
% The algorithm finds the optimal weights to the WB problem when the
% support is fixed. The algorithm is implemented according to Algorithm 1 
% in (Cuturi,2014).
% 
% Input:    X -- d x n matrix
%           Y_cell -- the N x 1 cell of d x m matrices Y_n
%           b_cell -- the N x 1 cell of weight vectors of Y_n
%           t0 -- the pre-defined value > 0 
%           tol -- stopping criterion
%           M_cell -- a N x 1 of n x m cost matrices
%           Max_Iter -- maximum number of iterations allowed
% Output:   a_hat -- optimal weight vector on X 
 
    if ~exist('tol', 'var') || isempty(tol)
         tol = 1e-6;
    end
    if ~exist('M_cell','var') || isempty(M_cell)
         % form all cost matrices
         M_cell = Compute_M(X, Y_cell);
    end
    if ~exist('Max_Iter', 'var') || isempty(Max_Iter)
        Max_Iter = 100;
    end
    N = max(size(Y_cell)); 
    n = size(X, 2);
           
    a_hat = ones(n ,1)/n;
    a_tilde = ones(n ,1)/n;
    t = 1;
    In = ones(n, 1);
    
    lambda = 1;
    % update a
    while (1)
        beta = (t + 1)/2;
        a = (1 - 1/beta) * a_hat + a_tilde./beta;
        
        % find dual optimal, the subgradient, via Sinkhorn's algorithm
        alpha = zeros(n, 1);
        for k = 1 : N
%             fprintf("Fixed support Algo : compute optimal solution of dual problem for sample %d\n", k)
            alpha = alpha + Wdist_Sinkhorn(M_cell{k}, a, b_cell{k}, lambda, tol, Max_Iter);
        end
        alpha = alpha / N;
        
        % optimal of proximal operator when the Bregman divergence
        % B is induced by the entropy function, i.e., it is KL divergence
        a_tilde = a_tilde .* (exp(-t0 * beta * alpha));
        a_tilde = a_tilde./(a_tilde' * In);
       
        a_hat = (1 - 1/beta) * a_hat + a_tilde./beta;
        t = t + 1;
        
        if (Euclid_dist(a_hat, a_tilde) < tol) || (t > Max_Iter)
            break;
        end
    end

end