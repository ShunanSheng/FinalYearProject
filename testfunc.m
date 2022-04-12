X = [1, 0, 2; 0 ,1, 2]
Y1 = [0, 1; 1.1, 0]
Y2 = [0, 1; 1, 0];
Y_cell = {Y1, Y2};

% Euclid_dist(X, Y)
% result = Compute_M(X, Y_cell)

diag(vecnorm(X).^2)
 x = diag(X' * X)

 t = 1;
 while(1)
     t = t + 1;
     if (t > 10)
         break;
     end
 end

 t