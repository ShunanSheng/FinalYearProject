function W2dist = Wdist_Gauss(N1, N2)
    % compute W2 distance between two normal distributions
    S1_sqrt = sqrtm(N1.Sig);
    W2dist = sum((N1.mu - N2.mu).^2) +  ...
        trace(N1.Sig + N2.Sig -2 * sqrtm(S1_sqrt * N2.Sig * S1_sqrt));
end