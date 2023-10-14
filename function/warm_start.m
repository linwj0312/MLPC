% function to warm start by using HOSVD, each slice of the core tensor C is
% calculated by the closed form solution.
function [A_init, B_init, C_init] = warm_start(X, Y, opts)
    [N,D] = size(X);
    [~,L] = size(Y);
    W = (X' * X + opts.lambda_A * eye(D)) \ (X' * Y);
    W_tensor = tensor(reshape(repmat(W',N+1,1), [L, N+1, D]));
    temp = hosvd(W_tensor, 1e-1, 'rank',[opts.r1, opts.r2, D]);
    A_init = temp.U{1} + eps * 1e5;
    B_init = temp.U{2} + eps * 1e5;
    C_init = ttm(temp.core, temp.U{3}, 3) + eps * 1e5;  
end