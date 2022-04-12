function [alpha_opt, varargout] = Wdist_Sinkhorn(M, a, b, lambda, tol, Max_Iter)
% Find the optimal solutions to the entropy-regularized primal-dual tranport 
% problems defined in Section 5.1 in (Cuturi,2014) using Sinkhorn's Algorithm.
% Input:        M -- the n x m cost matrix of X (d x n matrix) and Y (d x m matrix)
%               lambda -- the regularization parameter > 0
%               a -- the weight vector of X
%               b -- the weight vector of Y
%               tol -- error tolerance for stopping criterion
%               Max_Iter -- maximum number of iterations allowed
% Output: 
%               T_opt -- the primal optimal solution (p_ij), n x m matrix 
%               alpha_opt -- the dual optimal solution, n x 1 vector
% 
    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 1;
    end
    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-3;
    end
    if ~exist('Max_Iter', 'var') || isempty(Max_Iter)
        Max_Iter = 5000;
    end
    n = size(M, 1);
    
    K = exp(-lambda.* M);
    K(K<1e-200) = 1e-200;
    I=(a>0);
    if ~all(I) % need to update some vectors and matrices if a does not have full support
        K=K(I,:);
        a=a(I);
    end
    K_tilde = K./a;
    u = ones(n, 1)./n;
    
    % fixed-point iteration
    t = 1;
    while(t < Max_Iter)
        v = b./((u' * K)');
        u = 1./(K_tilde * v);
        t = t + 1;
        if (mod(t, 20) == 0)
            Criterion = norm(sum(abs(v.*(K'*u)-b)),Inf); % we use marginal difference criterion
%             fprintf("Sinkhorn algo : Iteration %d, with distance %6.4f\n", t, Criterion)
            if (Criterion < tol)|| isnan(Criterion)
                break;
            end
        end
    end
    
    In = ones(n, 1);
    alpha_opt = - log(u)./lambda + ((log(u)' * In)/(lambda * n)) * In;
    T_opt = diag(u) * K * diag(v);
    varargout{1} = T_opt;
    if nargout > 2
        varargout{2} = sum(sum(T_opt.* M));
    end
end