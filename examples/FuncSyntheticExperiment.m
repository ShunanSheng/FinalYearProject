function [MSloss, bar, opf, ipf, ibar, kf] = FuncSyntheticExperiment(nw, opf, kf_model, options)
    %% Given the system and observation models, simulate the results and
    % conduct the particle filters
    %% Extract parameters
    sys = nw.sys;
    obs = nw.obs;
    nx = nw.nx;
    ny = nw.ny;
    T = nw.T;
    M = nw.M;
    gen_sys_noise = nw.gen_sys_noise;
    gen_obs_noise = nw.gen_obs_noise;
    Ns = opf{1}.Ns;
    ipf = opf;
    %% Separate memory space
    x = zeros(nx,T);  y = zeros(M,ny,T);
    u = zeros(nx,T);  v = zeros(M,ny,T);
    %% Set up random seed
    if options.rnd
      rng("default");
    end

    %% Simulate system
    % rng("default")
    xh0 = zeros(nx, 1);                                  % initial state
    u(:,1) = zeros(nx, 1);                               % initial process noise
    x(:,1) = xh0;
    for m = 1:M
        v(m,:,1) = gen_obs_noise();          % initial observation noise
        y(m,:,1) = obs(1, xh0, v(m, :,1));
    end
    for k = 2:T
       % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
       u(:,k) = gen_sys_noise();              % simulate process noise
       x(:,k) = sys(k, x(:,k-1), u(:,k));     % simulate state
       for m = 1:M
           v(m,:,k) = gen_obs_noise();              % simulate observation noise
           y(m,:,k) = obs(k, x(:,k), v(m, :,k));     % simulate observation
       end
    end
    
    %% Estimate state using one-step particle filters (OPF)
    % Separate memory storage
    resampling_strategy = 'systematic_resampling';
    xh = zeros(M, nx, T); xh(:,:,1) = repmat(xh0',M,1);
    yh = zeros(M, ny, T); yh(:,:,1) = repmat(obs(1, xh0, 0), M, 1);
    
    if options.print
           fprintf('One-step particle filters\n');
    end
    for k = 2:T
       if options.print
           fprintf('Iteration = %d/%d\n',k,T);
       end
        % state estimation
       for m = 1 : M
           opf{m}.k = k;
           [xh(m,:,k), opf{m}] = particle_filter(sys, y(m,:,k), opf{m}, resampling_strategy);  
       end
       % filtered observation
       yh(m,:,k) = obs(k, reshape(xh(m,:,k), nx, 1), 0);
    end


    %% extract the end state information and fuse using a free support W2 barycenter
    n = Ns * M;
    t0 = 0.1;
    tol = 1e-6;
    Max_Iter = 200;
    Y_cell = cell(M,1); b_cell = cell(M,1);
    X_free = zeros(nx * T, n);
    a_free = zeros(n, 1);
    x_pred_temp = zeros(nx * T, 1);
    for m = 1:M
        Y_cell{m} = reshape(permute(opf{m}.particles, [1,3,2]),[], Ns);
        b_cell{m} = opf{m}.w(:,T);
    end
    [X_free, a_free] = WB_free_support(n, Y_cell, b_cell, t0, tol, Max_Iter);
    x_pred_temp = X_free *  a_free;
    x_pred = reshape(x_pred_temp, [], T);

    %% Estimate state using iterative particle filters (IPF)
    xhi = zeros(nx, T); xhi(:,1) = xh0;
    ibar.particles = zeros(nx, Ns, T);
    ibar.w = zeros(Ns,T);
    ibar.particles(:,:,1) = repmat(xh0, 1, Ns);
    ibar.w(:,1) = ones(Ns,1)./Ns;
    if options.print
           fprintf('Iterative particle filters\n');
    end
    for k = 2:T
       if options.print
           fprintf('Iteration = %d/%d\n',k,T);
       end
       Y_cell = cell(M, 1); b_cell = cell(M, 1);  % initialize the Y_cell and b_cell
        % state estimation at each sensor
       for m = 1 : M
           ipf{m}.k = k;
           [~, ipf{m}] = iterative_particle_filter(sys, y(m,:,k), ipf{m}, ibar, resampling_strategy);
           Y_cell{m} = ipf{m}.particles(:,:,k);
           b_cell{m} = ipf{m}.w(:,k);
       end
        [ibar.particles(:,:,k), ibar.w(:,k)] = WB_free_support(Ns, Y_cell, b_cell, t0, tol, Max_Iter);
       % filtered observation
       xhi(:,k) = ibar.particles(:,:,k) * ibar.w(:,k) ;
    end
    %% Use Kalman filter
    kf.m = cell(T,1);
    kf.P = cell(T,1);
    kf.k = 1;
    kf.m{1} = kf_model.m0;
    kf.P{1} = kf_model.P0;

    xkf = zeros(nx, T);
    y_tot = reshape(y, [], T);
    for k = 2:T
        kf.k = k;
        [xkf(:,k), kf] = Kalman_filter(kf_model, kf, y_tot(:,k));
    end
    
    %% store the results
    bar.X_free = X_free;
    bar.a_free = a_free;  
    
    %% calcultate the loss
    MSloss = zeros(M+3, 1);
    for m = 1:M
        MSloss(m) = sum((x - reshape(xh(m,:,:),[],T)).^2, 'all')/T;
    end
    MSloss(M+1) = sum((x - x_pred).^2 ,'all')/T;
    MSloss(M+2) = sum((x - xhi).^2, 'all')/T;
    MSloss(M+3) = sum((x - xkf).^2, 'all')/T;
    %% save, print, plot given options
    if options.print
        % print results
        fprintf("The mean-square loss of the trajectory:\n")
        disp(MSloss');
    end
    if options.save
        % save results
        save(options.file_name, 'x', 'bar', 'opf','kf','ibar','ipf')
    end
    if options.fig 
        % plot figures
        if nx == 1 % plot trajectory only when the state process is 1D
            close all
           
            figure('Position',[100,100,400,300]);
            tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
            S = {};
            plot(1:T, x,'-', 'LineWidth',4)
            S{end + 1} = 'true state';
            hold on 
            for m = 1:M
                plot(1:T, reshape(xh(m,:,:),[],1),'--', 'LineWidth',1)
                S{end+1} = strcat('S', num2str(m));
            end
            plot(1:T, x_pred, '-', 'LineWidth',4)
            hold off
            S{end+1} = 'OPF';
            title("Approximated Target State Process",'FontSize', 20)
            xlabel("Time",'FontSize', 15)
            ylabel("State",'FontSize',15)
            legend(S,'location', 'best')
            
            figure('Position',[100,100,400,300]);
            tight_subplot(1,1,[.01 .03],[.12 .08],[.10 .015])
            S = {};
            plot(1:T, x,'-', 'LineWidth',4)
            hold on 
            S{end + 1} = 'true state';
            plot(1:T, x_pred, '-', 'LineWidth',4)
            S{end+1} = 'OPF';
            plot(1:T, xhi, '-', 'LineWidth',4)
            S{end+1} = 'IPF';
            plot(1:T, xkf,'-k','LineWidth', 4)
            S{end+1} = 'Kalman filter';
            hold off
            title("Approximated Target State Process",'FontSize', 16)
            xlabel("Time",'FontSize', 15)
            ylabel("State",'FontSize',15)
            legend(S,'location', 'best')
           
        end
    end
end