clc, clear all, close all;
%% System model
%% Process equation x[k] = sys(k, x[k-1], u[k]);
%  we consider a Gaussian Linear model with x_k = x_{k-1} + u_k, where u_k
%  is an additive noise follwing N(0, sigma_u^2)
nx = 2;  % number of states [dimension of states]
sys = @(k, xkm1, uk) xkm1 + uk; % (returns column vector)

%% Observation equation y[k] = obs(k, x[k], v[k]);
%  y is a linear function of x, i.e., y_k = 3x_k + v_k,
%  also with an additive noise v ~ N(0, sigma_v^2)
ny = 1;% number of observations
a = [5, 5];
obs = @(k, xk, vk) a * xk + vk;                  % (returns column vector)

%% PDF of process noise and noise generator function
nu = nx;                                           % size of the vector of process noise
Sigma_u = 10 * eye(nx);
p_sys_noise   = @(u) mvnpdf(u, zeros(nx, 1), Sigma_u);
gen_sys_noise = @(u) mvnrnd(zeros(nx,1), Sigma_u)';         % sample from p_sys_noise (returns column vector)

%% PDF of observation noise and noise generator function
nv = ny;                                           % size of the vector of observation noise
sigma_v = sqrt(400);
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);      % pdf of zero mean
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)

%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) mvnrnd(zeros(1,nx), Sigma_u)';               % sample from p_x0 (returns column vector)

%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));

%% Number of time steps
T = 10;

%% Number of sensors  
M = 3;

%% Network model 
nw.M = M;                                
nw.T = T;
nw.sys = sys;
nw.obs = obs;
nw.nx = nx;
nw.ny = ny;
nw.gen_sys_noise = gen_sys_noise;
nw.gen_obs_noise = gen_obs_noise;

%% Build the particle filters parameters
pf.k               = 1;                   % initial iteration number
pf.Ns              = 100;                 % number of particles
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
options.fig = 0;
options.print = 1;
options.save = 1;
options.file_name = strcat("examples/Synthetic2DT=",num2str(T),".mat");
options.rnd = 0;

%% Build Kalman filter
model.F = eye(nx);
model.H = repmat(a, M ,1);
model.Q = Sigma_u;
model.R = diag(sigma_v.^2 * ones(M,1));
model.m0 = zeros(nx,1);
model.P0 = Sigma_u;

%% Run the function and calculate the MSE
[MSloss, bar, opf, ipf, ibar, kf] = FuncSyntheticExperiment(nw, opf, model, options);
save(options.file_name, 'MSloss','-append')
%%
T = 10;
options.file_name = strcat("examples/Synthetic2DT=",num2str(T),".mat");
s = load(options.file_name);
plotHist(bar,opf, ibar, kf, s.x, T)

%% Funtion utilities
function plotHist(bar, pfh, ibar, kf, x, k)
    close all;
    M = size(pfh,1);
    T = pfh{1}.k;
    if (size(bar.X_free, 1) == T) % i.e., nx = 1
        figure('Position',[100,100,400,300]);
        tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
        histogram(bar.X_free(T,:),'Normalization','Probability')
        hold on 
        for m = 1:M
            histogram(pfh{m}.particles(:,:,T),'Normalization','Probability')
        end
        hold off
    elseif (size(bar.X_free, 1) == 2 * T)  % i.e., nx =2
        X_free = bar.X_free((k-1) * 2 + 1: k * 2,:);
        figure('Position',[100,100,400,300]);
        tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
        S = {};
        kf_p = mvnrnd(kf.m{k}, kf.P{k}, 200)';
        scatter(kf_p(1,:), kf_p(2,:),50,'x','LineWidth',0.5);
        S{end+1} = "Kalman filter";
        hold on
%         hold on
%         for m = 1:M
%             particles = pfh{m}.particles(:,:,k);
%             scatter(particles(1,:), particles(2,:),10,'o');
%             S{end+1} = strcat("Sensor",num2str(m));
%         end
        scatter(X_free(1,:), X_free(2,:),80,'b^','filled','LineWidth',1.5);
        S{end+1} = "OPF";        
        X_ipf = ibar.particles(:,:,k);
        scatter(X_ipf(1,:),X_ipf(2,:), 80,'yv','filled','LineWidth',1.5)
        S{end+1} = "IPF";
        hold on
        scatter(x(1,k),x(2,k),80,'ro','filled');
        S{end+1} = "true state";
        hold off
        title(strcat("Distribution of End State, T=",num2str(k)),'FontSize', 20)
        legend(S, 'location', 'best');
    else
        error("Unable to plot particles with dimension greater or equal to 3")
    end
end



