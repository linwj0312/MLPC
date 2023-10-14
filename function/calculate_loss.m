% function to calculate loss
function loss = calculate_loss(A,B,C,Theta_A,Theta_B,Theta_C,X_tilde,Y,opts)
    W = ttm(C, {A,B}, [1 2]);
    C1 = tenmat(C,1).data;
    Theta_C_1 = tenmat(Theta_C,1).data;
    W_tilde = tenmat(W,1,'bc').data';
    loss = norm(Y - X_tilde * W_tilde,'fro')^2;
    switch opts.p
        case 1
            loss = loss + opts.lambda_A * norm(A ./ (Theta_A + eps),1) + opts.lambda_B * norm(B ./ (Theta_B + eps),1) + opts.lambda_C * norm(C1 ./ (Theta_C_1 + eps),1);
        case 2
            loss = loss + opts.lambda_A * norm(A ./ (Theta_A + eps),'fro')^2 + opts.lambda_B * norm(B ./ (Theta_B + eps),'fro')^2 + opts.lambda_C * norm(C1 ./ (Theta_C_1 + eps),'fro')^2;
    end
end

