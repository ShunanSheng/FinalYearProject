clc, clear all, close all;
%% System model
%% Process equation x[k] = sys(k, x[k-1], u[k]);
%  we consider a Gaussian Linear model with x_k = x_{k-1} + u_k, where u_k
%  is an additive noise follwing N(0, sigma_u^2)
nx = 1;  % number of states [dimension of states]
sys = @(k, xkm1, uk) xkm1 + uk; % (returns column vector)

%% Observation equation y[k] = obs(k, x[k], v[k]);
%  y is a linear function of x, i.e., y_k = 3x_k + v_k,
%  also with an additive noise v ~ N(0, sigma_v^2)
ny = 1;                                           % number of observations
obs = @(k, xk, vk) 5 * xk + vk;                  % (returns column vector)

%% PDF of process noise and noise generator function
nu = nx;                                           % size of the vector of process noise
sigma_u = sqrt(10);
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);         % sample from p_sys_noise (returns column vector)

%% PDF of observation noise and noise generator function
nv = ny;                                           % size of the vector of observation noise
sigma_v = sqrt(400);
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);      % pdf of zero mean
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) normrnd(0, sqrt(10));               % sample from p_x0 (returns column vector)

%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));

%% Number of time steps
T = 20;

%% Number of sensors  
M = 3;

%% Network model 
nw.M = M;                                
nw.T = T;
nw.sys = sys;
nw.obs = obs;
nw.nx = nx;
nw.ny = ny;
% nw.sigma_sys = sigma_u;
% nw.sigma_obs = sigma_v;
nw.gen_sys_noise = gen_sys_noise;
nw.gen_obs_noise = gen_obs_noise;

%% Build the particle filters parameters
pf.k               = 1;                   % initial iteration number
pf.Ns              = 200;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nx, pf.Ns, T); % particles
pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise

%% Build local particle filters paramters, here we assume all particle filters are the same
opf = cell(M,1);
for m = 1:M
    opf{m} = pf;
end
%% Print, plot, save options
options.fig = 1;
options.print = 1;
options.save = 0;
options.file_name = strcat("examples/Synthetic1DT=",num2str(T),".mat");
options.rnd = 1; 
%% Run the function and calculate the MSE
model.F = 1;
model.H = 5 * ones(M,1);
model.Q = sigma_u.^2;
model.R = diag(sigma_v.^2 * ones(M,1));
model.m0 = 0;
model.P0 = sqrt(10).^2;

% One trial
[MSloss, bar, opf, ipf, ibar, kf] = FuncSyntheticExperiment(nw, opf, model, options);
% %% Evaluate the mean-square error
% N = 20;
% MSloss_cell = cell(N, 1);
% for n = 1:N
%     [MSloss_cell{n}, ~, ~, ~, ~,~] = FuncSyntheticExperiment(nw, oph, model, options);
%     if mod(n, 5) ==0
%         fprintf("Iteration %d\n", n);
%     end
% end
% 
% MSloss = mean(cell2mat(MSloss_cell'), 2);
% save(options.file_name, 'MSloss','-append')

%% Funtion utilities
function plotHist(bar, opf)
    M = size(opf,1);
    T = size(bar.X_free, 3);
    figure('Position',[100,100,400,300]);
    tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
    histogram(bar.X_free(T,:),'Normalization','Probability')
    hold on 
    for m = 1:M
        histogram(opf{m}.particles(:,:,T),'Normalization','Probability')
    end
    hold off
end

