function [W_new,loss] = MLPC(X_train,X_Tilde,Y_train,opts)
% *************************************************************************
% Multi-Label Personalized Classification (MLPC)
% Input:
%      Training data (double): X_train, Y_train
%      Processed data (double): X_Tilde
%      Hyperparameters: opts
% Output:
%      Coefficient tensor (tensor): W_new
%      Objective values: loss
% *************************************************************************
    [N,D] = size(X_train);
    %% main loop of MLPC
    % warm start to help converge
    [A_init, B_init, C_init] = warm_start(X_train, Y_train, opts);
    A_old = A_init;
    B_old = B_init;
    C_old = C_init;
    Theta_A = calculate_Theta(opts,A_old);
    Theta_B = calculate_Theta(opts,B_old);
    Theta_C = calculate_Theta(opts,tenmat(C_old,1).data);
    Delta = sparse([ones(1,N);eye(N)]);
    iter = 0;
    loss = [];
    % main loop
    while iter < opts.max_outer_iter
        % 1. update A
        [A_new,~] = update_A_grad(A_old,B_old,C_old,Theta_A,X_train,X_Tilde,Y_train,Delta,opts);
        Theta_A = calculate_Theta(opts,A_new);
        A_old = A_new;
        % 2. update B
        [B_new,~] = update_B_grad(A_new,B_old,C_old,Theta_B,X_train,X_Tilde,Y_train,Delta,opts);
        Theta_B = calculate_Theta(opts,B_new);
        B_old = B_new;
        % 3. update C
        [C_new_matrix,C_new,~] = update_C_grad(A_new,B_new,C_old,Theta_C,X_train,X_Tilde,Y_train,Delta,opts);
        Theta_C = calculate_Theta(opts,C_new_matrix);
        C_old = C_new;
        % calculate loss
        loss = cat(1,loss, calculate_loss(A_new,B_new,C_new,Theta_A,Theta_B,Theta_C,X_Tilde,Y_train,opts));
        % convegence condition
        if iter >= 2 && abs(loss(end-1)-loss(end))/abs(loss(end-1)) <= opts.tol
            break;
        end
        iter = iter + 1;
    end

    W_new = ttm(tensor(reshape(Theta_C,[opts.r1, opts.r2, D])) .* C_new,{Theta_A .* A_new,Theta_B .* B_new},[1 2]);
end
%% private funtion to update Theta_A, Theta_B and Theta_C
function Theta_X = calculate_Theta(opts,X)
    norm_X = sum(abs(X).^opts.const1,2).^(1 / opts.const1) + eps;
    Theta_X = (bsxfun(@rdivide,abs(X),norm_X)).^opts.const2;
end
