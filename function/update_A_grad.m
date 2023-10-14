%% Function to update A by fixing B and C
function [A_new,A_loss] = update_A_grad(A,B,C,Theta_A,X,X_Tilde,Y,Delta,opts)
    [N,D] = size(X);
    iter_A = 0;
    A_old = A;
    A_loss = [];
    C1 = tenmat(C,1).data;
    F = zeros(opts.r1,N);
    for i = 1:N
        F(:,i) = C1 * kron(eye(D)' * X(i,:)',B' * Delta(:,i));
    end
    % main loop to update A
    while iter_A < opts.max_inner_iter
        grad_A = calculate_grad_A(A_old,Y,F,N);
        opts.learning_rate_A = 1;
        switch opts.p
            % p = 1, we use proximal gradient descend
            case 1
                % line search
                while true
                    A_current = soft_thresholding(A_old - opts.learning_rate_A * grad_A, opts);
                    if calculate_loss_A(A_current,B,C,Theta_A,X_Tilde,Y,opts) <= calculate_loss_A(soft_thresholding(A_old,opts),B,C,Theta_A,X_Tilde,Y,opts) - opts.learning_rate_A * norm(grad_A,'fro')^2 / 2
                        break;
                    else
                        opts.learning_rate_A = opts.learning_rate_A / 10;
                    end
                end
            % p = 2, we use gradient descend
            case 2
                grad_A = grad_A + 2 * opts.lambda_A * A_old ./ ((Theta_A).^2);
                % line search
                while true
                    A_current = A_old - opts.learning_rate_A * grad_A;
                    if calculate_loss_A(A_current,B,C,Theta_A,X_Tilde,Y,opts) <= calculate_loss_A(A_old,B,C,Theta_A,X_Tilde,Y,opts) - opts.learning_rate_A * norm(grad_A,'fro')^2 / 2
                        break;
                    else
                        opts.learning_rate_A = opts.learning_rate_A / 10;
                    end
                end
        end
        % calculate loss after updating A
        A_loss = cat(1,A_loss, calculate_loss_A(A_current,B,C,Theta_A,X_Tilde,Y,opts));
        % check whether the algorithm convegers
        if iter_A >= 2 && abs(A_loss(end-1) - A_loss(end)) / abs(A_loss(end-1)) <= opts.tol
            break;
        end
        iter_A = iter_A + 1;
        A_old = A_current;
    end
    A_new = A_current;
end
%% private function to calculate gradient of A
function grad_A = calculate_grad_A(A,Y,F,N)
    grad_A = zeros(size(A));
    for i = 1:N
        grad_A = grad_A + 2 * (A * F(:,i) - Y(i,:)') * F(:,i)';
    end
end
%% private function to calculate loss of A
function loss = calculate_loss_A(A,B,G,Theta_A,X_Tilde,Y,opts)
    W = ttm(G, {A,B}, [1 2]);
    W_tilde = tenmat(W,1,'bc').data';
    loss = norm(Y - X_Tilde * W_tilde,'fro')^2;
    switch opts.p
        case 1
            loss = loss + opts.lambda_A * norm(A ./ (Theta_A + eps),1);
        case 2
            loss = loss + opts.lambda_A * norm(A ./ (Theta_A + eps),'fro')^2;
    end    
end
%% Soft Thresholding function
function X = soft_thresholding(A, opts)
    X = sign(A) .* max(0,abs(A) - opts.lambda_A);
end