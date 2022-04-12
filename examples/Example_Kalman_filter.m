clear all, close all, clc
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
obs = @(k, xk, vk) 1 * xk + vk;                  % (returns column vector)

%% PDF of process noise and noise generator function
nu = nx;                                           % size of the vector of process noise
sigma_u = sqrt(1);
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);         % sample from p_sys_noise (returns column vector)

%% PDF of observation noise and noise generator function
nv = ny;                                           % size of the vector of observation noise
sigma_v = sqrt(1);
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);      % pdf of zero mean
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) normrnd(0, sqrt(1));               % sample from p_x0 (returns column vector)

%% Number of time steps
T = 200;

%% Simulate system
xh0 = 0;                                  % initial state
u(:,1) = 0;                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = xh0;
y(:,1) = obs(1, xh0, v(:,1));
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   u(:,k) = gen_sys_noise();              % simulate process noise
   v(:,k) = gen_obs_noise();              % simulate observation noise
   x(:,k) = sys(k, x(:,k-1), u(:,k));     % simulate state
   y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
end


%% Use Kalman filter
model.F = 1;
model.H = 1;
model.Q = sigma_u.^2;
model.R = sigma_v.^2;
model.m0 = 0;
model.P0 = sqrt(1).^2;

kf.m = cell(T,1);
kf.P = cell(T,1);
kf.k = 1;

kf.m{1} = model.m0;
kf.P{1} = model.P0;

xkf = zeros(nx, T);

for k = 2:T
    kf.k = k;
    [xkf(:,k), kf] = Kalman_filter(model, kf, y(:,k));
end

figure()
plot(1:T, x, '-','LineWidth', 4)
hold on
plot(1:T, xkf,'--', 'LineWidth', 4)

figure()
plot(1:T, cell2mat(kf.P), '-','LineWidth', 4)




