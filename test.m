%     X_free = zeros(nx,n,T);
%     a_free = zeros(n,T);    
%     x_pred = zeros(nx,T);
%     for k = 1:T
%         Y_cell = cell(M, 1); b_cell = cell(M, 1);  % initialize the Y_cell and b_cell
%         for m = 1:M
%             Y_cell{m} = pfh{m}.particles(:,:,k);
%             if k == 1
%                 b_cell{m} = ones(Ns, 1)./(Ns);
%             else
%                 b_cell{m} = pfh{m}.w(:,k);
%             end
%         end
%         [X_free(:,:,k), a_free(:,k)] = WB_free_support(n, Y_cell, b_cell, t0, tol, Max_Iter);
%         x_pred(:, k) = X_free(:,:,k) * a_free(:,k);
%         if mod(k,10) == 0
%             fprintf("Iteration %d : true state is %4.2f, estimated state is %4.2f\n", k,...
%                 x(k),x_pred(:,k));
%         end
%     end