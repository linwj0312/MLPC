%% Function to update C by fixing A and B
function [C_new_matrix,C_new,C_loss] = update_C_grad(A,B,C,Theta_C,X,X_Tilde,Y,Delta,opts)
    [N,D] = size(X);
    iter_C = 0;
    C_old = tenmat(C,1).data;
    C_loss = [];
    F = zeros(opts.r2 * D,N);
    for i = 1:N
        F(:,i) = kron(eye(D)' * X(i,:)',B' * Delta(:,i));
    end
    % main loop to update C
    while iter_C < opts.max_inner_iter
        grad_C = calculate_grad_C(A,C_old,Y,F,N);
        opts.learning_rate_C = 1;
        switch opts.p
            % p = 1, we use proximal gradient descend
            case 1
                % line search
                while true
                    C_current = soft_thresholding(C_old - opts.learning_rate_C * grad_C, opts);
                    if calculate_loss_C(A,B,C_current,Theta_C,X_Tilde,Y,opts,D) <= calculate_loss_C(A,B,soft_thresholding(C_old,opts),Theta_C,X_Tilde,Y,opts,D) - opts.learning_rate_C * norm(grad_C,'fro')^2 / 2
                        break;
                    else
                        opts.learning_rate_C = opts.learning_rate_C / 10;
                    end
                end
            % p = 2, we use gradient descend
            case 2
                grad_C = grad_C + 2 * opts.lambda_C * C_old ./ ((Theta_C).^2);
                % line search
                while true
                    C_current = C_old - opts.learning_rate_C * grad_C;
                    if calculate_loss_C(A,B,C_current,Theta_C,X_Tilde,Y,opts,D) <= calculate_loss_C(A,B,C_old,Theta_C,X_Tilde,Y,opts,D) - opts.learning_rate_C * norm(grad_C,'fro')^2 / 2
                        break;
                    else
                        opts.learning_rate_C = opts.learning_rate_C / 10;
                    end
                end
        end
        % calculate loss after updating C
        C_loss = cat(1,C_loss, calculate_loss_C(A,B,C_current,Theta_C,X_Tilde,Y,opts,D));
        % check whether the algorithm convegers
        if iter_C >= 2 && abs(C_loss(end-1) - C_loss(end)) / abs(C_loss(end-1)) <= opts.tol
            break;
        end
        iter_C = iter_C + 1;
        C_old = C_current;
    end
    C_new_matrix = C_current;
    C_new = tensor(reshape(C_current,[opts.r1,opts.r2,D]));
end
%% private function to calculate gradient of C
function grad_C = calculate_grad_C(A,C,Y,F,N)
    grad_C = zeros(size(C));
    for i = 1:N
        grad_C = grad_C + 2 * A' * (A * C * F(:,i) - Y(i,:)') * F(:,i)';
    end
end
%% private function to calculate loss of C
function loss = calculate_loss_C(A,B,C,Theta_C,X_Tilde,Y,opts,D)
    C_tensor = tensor(reshape(C,[opts.r1,opts.r2,D]));
    W = ttm(C_tensor, {A,B}, [1 2]);
    W_tilde = tenmat(W,1,'bc').data';
    loss = norm(Y - X_Tilde * W_tilde,'fro')^2;
    switch opts.p
        case 1
            loss = loss + opts.lambda_C * norm(C ./ (Theta_C + eps),1);
        case 2
            loss = loss + opts.lambda_C * norm(C ./ (Theta_C + eps),'fro')^2;
    end    
end
%% Soft Thresholding function
function X = soft_thresholding(C, opts)
    X = sign(C) .* max(0,abs(C) - opts.lambda_G);
end