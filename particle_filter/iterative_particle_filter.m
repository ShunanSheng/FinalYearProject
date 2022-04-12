function [xhk, ipf] = iterative_particle_filter(sys, yk, ipf, ibar, resampling_strategy)
%% Iterative particle filter
% Note: compared to the general particle filter, the previous step
% particles are W2 barycenter 
%
% Usage:
% [xhk, ipf] = iterative_particle_filter(sys, yk, ipf, ibar, resamping_strategy)
%
% Inputs:
% sys  = function handle to process equation
% yk   = observation vector at time k (column vector)
% ipf   = structure with the following fields
%   .k                = iteration number
%   .Ns               = number of particles
%   .w                = weights   (Ns x T)
%   .particles        = particles (nx x Ns x T)
%   .gen_x0           = function handle of a procedure that samples from the initial pdf p_x0
%   .p_yk_given_xk    = function handle of the observation likelihood PDF p(y[k] | x[k])
%   .gen_sys_noise    = function handle of a procedure that generates system noise
% ibar = structure with the following fileds
%   .w                = weights   (Ns x T), correspondin weights for the
%                                           particles
%   .particles        = particles (nx x Ns x T), the particles in the W2
%                       barycenter
% resampling_strategy = resampling strategy. Set it either to 
%                       'multinomial_resampling' or 'systematic_resampling'
%
% Outputs:
% xhk   = estimated state
% ipf    = the same structure as in the input but updated at iteration k


%%
k = ipf.k;
if k == 1
   error('error: k must be an integer greater or equal than 2');
end

%% Initialize variables
Ns = ipf.Ns;                              % number of particles
nx = size(ipf.particles,1);               % number of states

if k == 2
   for i = 1:Ns                          % simulate initial particles
      ipf.particles(:,i,1) = ipf.gen_x0(); % at time k=1
   end 
   wkm1 = repmat(1/Ns, Ns, 1);           % all particles have the same weight
   ipf.w(:,k) = wkm1;
end

%% Separate memory
xkm1 = ibar.particles(:,:,k-1); % extract particles from last iteration;
wkm1 = ibar.w(:,k-1);                     % weights of last iteration
xk   = zeros(size(xkm1));     % = zeros(nx,Ns);
wk   = zeros(size(wkm1));     % = zeros(Ns,1);

%% Algorithm 3 of Ref [1]
for i = 1:Ns
   % xk(:,i) = sample_vector_from q_xk_given_xkm1_yk given xkm1(:,i) and yk
   % Using the PRIOR PDF: pf.p_xk_given_xkm1: eq 62, Ref 1.
   xk(:,i) = sys(k, xkm1(:,i), ipf.gen_sys_noise());
   
   % Equation 48, Ref 1.
   % wk(i) = wkm1(i) * p_yk_given_xk(yk, xk(:,i))*p_xk_given_xkm1(xk(:,i), xkm1(:,i))/q_xk_given_xkm1_yk(xk(:,i), xkm1(:,i), yk);
   
   % weights (when using the PRIOR pdf): eq 63, Ref 1
   wk(i) = wkm1(i) * ipf.p_yk_given_xk(k, yk, xk(:,i));
   
   % weights (when using the OPTIMAL pdf): eq 53, Ref 1
   % wk(i) = wkm1(i) * p_yk_given_xkm1(yk, xkm1(:,i)); % we do not know this PDF
end

%% Normalize weight vector
wk = wk./sum(wk);

%% Calculate effective sample size: eq 48, Ref 1
Neff = 1/sum(wk.^2);

%% Resampling
% remove this condition and sample on each iteration:
% [xk, wk] = resample(xk, wk, resampling_strategy);
%if you want to implement the bootstrap particle filter
resample_percentaje = 0.50;
Nt = resample_percentaje * Ns;
if Neff < Nt
   [xk, wk] = resample(xk, wk, resampling_strategy);
   % {xk, wk} is an approximate discrete representation of p(x_k | y_{1:k})
end

%% Compute estimated state
xhk = zeros(nx,1);
for i = 1:Ns
   xhk = xhk + wk(i) * xk(:,i); % compute weighted average
end
%% Store new weights and particles
ipf.w(:,k) = wk;
ipf.particles(:,:,k) = xk;
end