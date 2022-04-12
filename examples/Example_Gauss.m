clc, clear all, close all
% we expriemnt with the Normal distributions
% Sigma1 = [2, 1; 1, 2];
% Sigma2 = [3, 2; 2, 3];
% Sigma3 = [10, 0.5; 0.5, 100];

d = 2;
Sigma1 = 5 * eye(d);
Sigma2 = 5 * eye(d);
Sigma3 = 5 * eye(d);

N1.mu = zeros(d, 1) + 1;
N1.Sig = Sigma1;

N2.mu = zeros(d, 1) + 5;
N2.Sig = Sigma2;

N3.mu = zeros(d, 1) + 3;
N3.Sig = Sigma3;

%% Compute exact WB using fixed point iteration

input_cell = {N1, N2, N3};
lambda = [1/3, 1/3, 1/3];
tol = 1e-10;

[WB, cost] = WB_Gauss(input_cell, lambda, tol);

%% Compute approximate WB from samples using algorithm 2 in (Cuturi, 2014).
M = 3;
m = 100;
n = m;
N1.Sigma_chol = chol(Sigma1);
N2.Sigma_chol = chol(Sigma2);
N3.Sigma_chol = chol(Sigma3);
Y1 = N1.mu + N1.Sigma_chol' * randn(d,m);
Y2 = N2.mu + N2.Sigma_chol' * randn(d,m);
Y3 = N3.mu + N3.Sigma_chol' * randn(d,m);

Im = ones(m,1)/m;
Y_cell = {Y1,Y2,Y3}';
b_cell = {Im, Im, Im}';
t0 = 0.1;
tol = 1e-6;
Max_Iter = 5000;
lambda = 10;
%% Sinkhorn
b = Im;
M = Euclid_dist(Y1, Y2);
M = M./median(M(:));
[~,D] = Wdist_LP(M, b, b);
display(D)
%% the original implementation
K = exp(-lambda*M);
K(K<1e-100) = 1e-100;
U = K.*M;
a = b;

%% 
lambda_lst = [0.01,0.1:0.1:1, 1:1:10, 10:10:100]; L = length(lambda_lst);
W2Val_lst = zeros(L, 1);
W2valOri_lst = zeros(L,1);
for i = 1:L
[~, ~, W2Val_lst(i)] = Wdist_Sinkhorn(M, a, b, lambda_lst(i) , tol, Max_Iter);
[W2valOri_lst(i),~,~,~] = sinkhornTransport(a,b,K,U,lambda_lst(i),[],[],[],[],0);
end

close all
figure()
plot(lambda_lst, D * ones(L,1),'r','LineWidth',4)
hold on
plot(lambda_lst, W2Val_lst, 'b', 'LineWidth', 4)
hold on 
plot(lambda_lst, W2valOri_lst, 'g', 'LineWidth', 4)
legend("true","my implemenation","cuturi")

%% free support
lambda = 20;
[X_free, a_free] = WB_free_support(n, Y_cell, b_cell, t0, tol, Max_Iter);

M = Euclid_dist(X_free, Y3);
b = Im;
[~, ~, W2Val_free] = Wdist_Sinkhorn(M, a_free, b, lambda , tol, Max_Iter);

%% fixed support
s = 3/2/sqrt(2);
S = linspace(-s, s, 10);
[x1, x2] = meshgrid(S,S);
X1 = reshape(x1, [], 1);
X2 = reshape(x2, [], 1);
X_fixed = [X1' ; X2'];

M_cell = Compute_M(X_fixed, Y_cell);
a_fixed = WB_fixed_support(X_fixed, Y_cell, b_cell, t0, tol, M_cell, Max_Iter);

M = M_cell{1};
b = Im;
[~, ~, W2Val_fixed] = Wdist_Sinkhorn(M, a_fixed, b, lambda , tol, Max_Iter)

%% Sinkhorn in original implementation
% set lambda
Max_Iter = 200;
lambda = 20;

% in practical situations it might be a good idea to do the following:
%K(K<1e-100)=1e-100;
K = exp(-lambda*M);
% pre-compute matrix U, the Schur product of K and M.
K(K<1e-100) = 1e-100;
U = K.*M;

a = b;
[D,L,u,v] = sinkhornTransport(a,b,K,U,lambda,[],[],[],[],1);
fprintf("Result form original implementation is %4.2f\n", D)
[~, T, W2Val_lst] = Wdist_Sinkhorn(M, b, b, lambda , tol, Max_Iter);
fprintf("Result form original implementation is %4.2f\n", W2Val_lst)
T_original = diag(u) * K * diag(v);
sum(sum(abs(T - T_original)))

%% same results
file_name = "examples/Gauss.mat"
save(file_name, 'Y_cell','b_cell', 'X_free', 'a_free', 'W2Val_free', 'X_fixed', 'a_fixed', 'W2Val_fixed')

