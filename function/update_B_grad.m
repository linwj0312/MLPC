%% Function to update B by fixing A and C
function [B_new,B_loss] = update_B_grad(A,B,C,Theta_B,X,X_Tilde,Y,Delta,opts)
    [N,D] = size(X);
    iter_B = 0;
    B_old = B;
    B_loss = [];
    C2 = tenmat(C,2).data;
    F = cell(N,1);
    for i = 1:N
        F{i} = C2 * kron(eye(D)' * X(i,:)',A');
    end
    % main loop to update B
    while iter_B < opts.max_inner_iter
        grad_B = calculate_grad_B(B_old,Delta,Y,F,N);
        opts.learning_rate_B = 1;
        switch opts.p
            % p = 1, we use proximal gradient descend
            case 1
                % line search
                while true
                    B_current = soft_thresholding(B_old - opts.learning_rate_B * grad_B, opts);
                    if calculate_loss_B(A,B_current,C,Theta_B,X_Tilde,Y,opts) <= calculate_loss_B(A,soft_thresholding(B_old,opts),C,Theta_B,X_Tilde,Y,opts) - opts.learning_rate_B * norm(grad_B,'fro')^2 / 2
                        break;
                    else
                        opts.learning_rate_B = opts.learning_rate_B / 10;
                    end
                end
            % p = 2, we use gradient descend
            case 2
                grad_B = grad_B + 2 * opts.lambda_B * B_old ./ ((Theta_B).^2);
                % line search
                while true
                    B_current = B_old - opts.learning_rate_B * grad_B;
                    if calculate_loss_B(A,B_current,C,Theta_B,X_Tilde,Y,opts) <= calculate_loss_B(A,B_old,C,Theta_B,X_Tilde,Y,opts) - opts.learning_rate_B * norm(grad_B,'fro')^2 / 2
                        break;
                    else
                        opts.learning_rate_B = opts.learning_rate_B / 10;
                    end
                end
        end
        % calculate loss after updating B
        B_loss = cat(1,B_loss, calculate_loss_B(A,B_current,C,Theta_B,X_Tilde,Y,opts));
        % check whether the algorithm convegers
        if iter_B >= 2 && abs(B_loss(end-1) - B_loss(end)) / abs(B_loss(end-1)) <= opts.tol
            break;
        end
        iter_B = iter_B + 1;
        B_old = B_current;
    end
    B_new = B_current;
end
%% private function to calculate gradient of B
function grad_B = calculate_grad_B(B,Delta,Y,F,N)
    grad_B = zeros(size(B));
    for i = 1:N
        grad_B = grad_B + 2 * Delta(:,i) * (Delta(:,i)' * B * F{i} - Y(i,:)) * F{i}';
    end
end
%% private function to calculate loss of B
function loss = calculate_loss_B(A,B,C,Theta_B,X_Tilde,Y,opts)
    W = ttm(C, {A,B}, [1 2]);
    W_tilde = tenmat(W,1,'bc').data';
    loss = norm(Y - X_Tilde * W_tilde,'fro')^2;
    switch opts.p
        case 1
            loss = loss + opts.lambda_B * norm(B ./ (Theta_B + eps),1);
        case 2
            loss = loss + opts.lambda_B * norm(B ./ (Theta_B + eps),'fro')^2;
    end    
end
%% Soft Thresholding function
function X = soft_thresholding(B, opts)
    X = sign(B) .* max(0,abs(B) - opts.lambda_B);
end