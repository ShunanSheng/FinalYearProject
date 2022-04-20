function [X, a] = WB_free_support(n, Y_cell, b_cell, t0, tol, Max_Iter)
% The algorithm finds the optimal weights to the WB problem when the
% support is free. The algorithm is implemented according to Algorithm 2 
% in (Cuturi,2014).
% 
% Input:    n -- the size of support
%           Y_cell -- the N x 1 cell of d x m matrices Y_n
%           b_cell -- the N x 1 cell of m x 1 weight vectors of Y_n
%           t0 -- the pre-defined value > 0 
%           tol -- stopping criterion
%           Max_Iter -- maximum number of iterations allowed
% Output:   a_hat -- optimal weight vector on X 
    
    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-6;
    end
    if ~exist('Max_Iter', 'var') || isempty(Max_Iter)
        Max_Iter = 100;
    end
    
    theta = 0.5;
    N = max(size(Y_cell));
    d = size(Y_cell{1}, 1);
    lambda = 10;
    
    X = randn(d, n);
    a = ones(n ,1)/n;
    T_cell = cell(N, 1);
    count = 1;
    while (count <= Max_Iter)
        a_iter = a;
        X_iter = X;
        % normalize the matrix 
        M_cell = Compute_M(X_iter, Y_cell);
        M_cell = NormalizeM(M_cell);
        a = WB_fixed_support(X_iter, Y_cell, b_cell, t0, tol, M_cell,Max_Iter);
        s = zeros(d, n); % the sum of all optimal transport matrices
        for k = 1 : N      
%           fprintf("Free support Algo : compute optimal transport for sample %d\n", k)
            % use LP computation
%             [T_cell{k},~] = Wdist_LP(M_cell{k}, a, b_cell{k});

            [~, T_cell{k}] = Wdist_Sinkhorn(M_cell{k}, a, b_cell{k}, lambda, tol, Max_Iter);

%             fprintf("Use original implementation\n")
%             K = exp(-lambda .* M_cell{k});
%             K(K<1e-200) = 1e-200;
%             U = K.*M_cell{k};
%             [~,~,u,v] = sinkhornTransport(a,b_cell{k},K,U,lambda,[],[],[],[],0);
%             T_cell{k} = diag(u) * K * diag(v);
            s = s + Y_cell{k} * T_cell{k}';
        end
        s = s./N; % take average
        
        X =  (1 - theta) * X_iter + theta * s * diag(1./a); 
        count = count + 1;
        if (max(abs(X_iter - X), [], 'all') < 1e-6) && (Euclid_dist(a, a_iter) < tol)
            break;
        end
        if mod(count, 100) == 0
%             fprintf("Free Support Algo: iteration %d with max error %6.4f\n", count, max((X_iter - X).^2, [], 'all'));
%             fprintf("...................weight euclidean error %6.4f\n",Euclid_dist(a, a_iter))
        end
    end
end