%% function to demonstrate an experiment of MLPC on a synthetic dataset with p = 2 and q = 2
close all
clear all;
clc;
mkdir('./image');
tic
%% set the hyperparameters
opts = [];
opts.max_inner_iter = 1000;
opts.max_outer_iter = 100;
opts.tol = 1e-4;
opts.p = 2;
opts.q = 2;
opts.const1 = (opts.p * opts.q) / (opts.p + opts.q);
opts.const2 = opts.p / (opts.p + opts.q);
opts.learning_rate_A = 1;
opts.learning_rate_B = 1;
opts.learning_rate_C = 1;
opts.r1 = 11;
opts.r2 = 12;
opts.lambda_A = 1e-3;
opts.lambda_B = 1e-3;
opts.lambda_C = 1e-3;
%% generate the synthetic dataset
N = 39;
D = 40;
L = 40;

A = zeros(L, opts.r1);
A(1:8,1:3) = normrnd(0, 1, [8,3]);
A(9:16,3:5) = normrnd(0, 1, [8,3]);
A(17:24,5:7) = normrnd(0, 1, [8,3]);
A(25:32,7:9) = normrnd(0, 1, [8,3]);
A(33:40,9:11) = normrnd(0, 1, [8,3]);
A = A + eps * 1e5;

B = zeros(N+1, opts.r2);
B(1:8,1:2) = normrnd(0, 1, [8,2]);
B(1:8,11:12) = normrnd(0, 1, [8,2]);
B(9:16,3:4) = normrnd(0, 1, [8,2]);
B(9:16,9:10) = normrnd(0, 1, [8,2]);
B(17:24,5:8) = normrnd(0, 1, [8,4]);
B(25:32,3:4) = normrnd(0, 1, [8,2]);
B(25:32,9:10) = normrnd(0, 1, [8,2]);
B(33:40,1:2) = normrnd(0, 1, [8,2]);
B(33:40,11:12) = normrnd(0, 1, [8,2]);
B = B + eps * 1e5;

C = zeros(opts.r1,opts.r2,D);
c = [diag(normrnd(0, 1, [opts.r1, 1])), zeros(opts.r1,1)] + [zeros(opts.r1,1), diag(normrnd(0, 1, [opts.r1, 1]))] + [diag(normrnd(0, 1, [opts.r1-1, 1]),-1), zeros(opts.r1,1)];

for i = 1:D
    C(:,:,i) = c + eps * 1e5;
end
C = tensor(C);
W = ttm(C, {A,B}, [1 2]);

X = normrnd(0,1,[N,D]);
X_Tilde = data_preprocess(X);
Y = zeros(N,L);
for i = 1:N
    Y(i,:) = (W(:,1,:).data + W(:,i+1,:).data) * transpose(X(i,:));
end
Y = Y + normrnd(0,0.1,[N, L]);
%% main loop 
A_old = A + normrnd(0, 0.1, [L, opts.r1]);
B_old = B + normrnd(0, 0.1, [N + 1, opts.r2]);
C_old = C + normrnd(0, 0.1, [opts.r1, opts.r2, D]);
Theta_A = calculate_Theta(opts,A_old);
Theta_B = calculate_Theta(opts,B_old);
Theta_C = calculate_Theta(opts,tenmat(C_old,1).data);
Delta = sparse([ones(1,N);eye(N)]);
iter = 0;
loss = [];
% main loop to update A, B and C
while iter < opts.max_outer_iter
    [A_new,~] = update_A_grad(A_old,B_old,C_old,Theta_A,X,X_Tilde,Y,Delta,opts);
    Theta_A = calculate_Theta(opts,A_new);
    A_old = A_new;

    [B_new,~] = update_B_grad(A_new,B_old,C_old,Theta_B,X,X_Tilde,Y,Delta,opts);
    Theta_B = calculate_Theta(opts,B_new);
    B_old = B_new;

    [C_new_matrix,C_new,~] = update_C_grad(A_new,B_new,C_old,Theta_C,X,X_Tilde,Y,Delta,opts);
    Theta_C = calculate_Theta(opts,C_new_matrix);
    C_old = C_new;

    loss = cat(1,loss, calculate_loss(A_new,B_new,C_new,Theta_A,Theta_B,Theta_C,X_Tilde,Y,opts));
    if iter >= 2 && abs(loss(end-1)-loss(end))/abs(loss(end-1)) <= opts.tol
        break;
    end
    iter = iter + 1;
end

W_new = ttm(tensor(reshape(Theta_C,[opts.r1, opts.r2, D])) .* C_new,{Theta_A .* A_new,Theta_B .* B_new},[1 2]);
toc
%% display results for A
fig1 = figure(1);
subplot(2,1,1);
imagesc(abs(A) ./ max(max(abs(A))));
colormap(flipud(gray));
set(gca,'Fontsize',20);
xlabel('$\mathbf{A}^{\ast}$','interpreter','latex');
colorbar;
subplot(2,1,2);
imagesc(abs(A_new .* Theta_A) ./ max(max(abs(A_new .* Theta_A))));
colormap(flipud(gray));
set(gca,'Fontsize',20);
xlabel('$\mathbf{A}$','interpreter','latex');
colorbar;
set(gcf, 'Position',  [100, 100, 400, 700])
saveas(fig1,['./image/Synthetic A ',num2str(opts.lambda_A),'_',num2str(opts.lambda_C),'.png'])
%% display results for B
fig2 = figure(2);
subplot(2,1,1);
imagesc(abs(B) ./ max(max(abs(B))));
colormap(flipud(gray));
set(gca,'Fontsize',20);
xlabel('$(\mathbf{B}^{\ast})^{T}$','interpreter','latex');
colorbar;
subplot(2,1,2);
imagesc(abs(B_new .* Theta_B) ./ max(max(abs(B_new .* Theta_B))));
colormap(flipud(gray));
set(gca,'Fontsize',20);
xlabel('$\mathbf{B}^{T}$','interpreter','latex');
colorbar;
set(gcf, 'Position',  [100, 100, 400, 700])
saveas(fig2,['./image/Synthetic B ',num2str(opts.lambda_A),'_',num2str(opts.lambda_C),'.png'])
%% display results for C
fig3 = figure(3);
subplot(2,1,1);
imagesc(abs(double(C(:,:,1))) ./ max(max(abs(double(C(:,:,1))))));
colormap(flipud(gray));
set(gca,'Fontsize',20);
xlabel('$\mathbf{C}_{:1:}^{\ast}$','interpreter','latex');
colorbar;
subplot(2,1,2);
temp1 = C_new .* reshape(Theta_C,[opts.r1, opts.r2, D]);
imagesc(abs(temp1(:,:,1).data) ./ max(max(abs(temp1(:,:,1).data))));
colormap(flipud(gray));
set(gca,'Fontsize',20);
xlabel('$\mathbf{C}_{:1:}$','interpreter','latex');
colorbar;
set(gcf, 'Position',  [100, 100, 400, 700])
saveas(fig3,['./image/Synthetic C ',num2str(opts.lambda_A),'_',num2str(opts.lambda_C),'.png'])
%% private function to calculate Theta
function Theta_X = calculate_Theta(opts,X)
    norm_X = sum(abs(X).^opts.const1,2).^(1 / opts.const1) + eps;
    Theta_X = (bsxfun(@rdivide,abs(X),norm_X)).^opts.const2;
end