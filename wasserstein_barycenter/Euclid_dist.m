function M  = Euclid_dist(X, Y)
% Construct the cost matrix M_XY given two disctre measures X, Y 
% with n,m points respectively. Asssume the space if Euclidean with
% d>=1 and p=2. Then 
%           M = x1(m)^T + 1(n)y^T - 2X^TY,
% where x = diag(X^T X), y = diag(Y^T Y), 1(n) is ones(n,1).
% Input:    X -- d x n matrix  
%           Y -- d x m matrix
% Output:   M -- n x m matrix 
    n = size(X, 2);
    m = size(Y, 2);
    if (size(X, 1) > 1) && (size(Y, 1) > 1)
        x = (vecnorm(X).^2)';
        y = (vecnorm(Y).^2)';
    else
        x = (X.^2)';
        y = (Y.^2)';
    end
    
    Im = ones(m, 1);
    In = ones(n, 1);
    
    M = x*Im' + In*y' - 2*X'*Y;
end
