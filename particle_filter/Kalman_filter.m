function [xh, kf] = Kalman_filter(model, kf, yk)
    % Given the system model and the observations, producing mean (mk) and
    % covariance (Pk) estimates of the posterior distribution, which is Gaussian
    % if the system model is a Gaussian linear model.
    % X_k = F * X_{k-1} + v, v ~ N(0, Q)
    % Y_k = H * X_k + n, n ~ N(0, R)
    F = model.F;
    H = model.H;
    Q = model.Q;
    R = model.R;
    
    k = kf.k;
    if k == 1
       error('error: k must be an integer greater or equal than 2');
    end
    mkm1 = kf.m{k-1};
    Pkm1 = kf.P{k-1};
    
    mk1 = F * mkm1;
    Pk1 = Q + F * Pkm1 * F';
    
    Sk = H * Pk1 * H' + R;
    Kk = Pk1 * H' / Sk;
    
    mk = mk1 + Kk * (yk - H * mk1);
    Pk = Pk1 - Kk * H * Pk1;
    
    kf.m{k} = mk;
    kf.P{k} = Pk;
    
    xh = mk;
end