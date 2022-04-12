function M_cell = Compute_M(X, Y_cell)
% Compute the cost matrices between X and a cell of Y
% Input:    X -- d x n matrix
%           Y_cell -- the N x 1 cell of d x m matrices Y_n
% Output:   M_cell -- a N x 1 of n x m cost matrices
    N = max(size(Y_cell));
    m = size(Y_cell{1}, 2);
    Y_all = horzcat(Y_cell{:});
    M_all  = Euclid_dist(X, Y_all);
    M_cell = cell(N, 1);
    for k = 1 : N
        M_cell{k} = M_all(:, (k-1) * m + 1 : k * m);
    end    
end