close all
clear all
clc

mkdir('./plot');

tic;
rng('default');
% load data
dataset = './data/yeast_new_1.mat';
load(dataset);
% data preprocess
[N_train, L] = size(trainSet.Y);
D = size(trainSet.X,2) + 1;
N_valid = size(validSet.X,1);
X_train = [trainSet.X,ones(N_train,1)];
Y_train = trainSet.Y;
X_valid = [validSet.X,ones(N_valid,1)];
Y_valid = validSet.Y;
X_train_process = data_preprocess(X_train);
X_valid_process = data_preprocess(X_valid);
%% hyperparameters
opts = [];
opts.max_inner_iter = 1000;
opts.max_outer_iter = 100;
opts.tol = 1e-4;
opts.r1 = 13;
opts.r2 = 20;
opts.p = 2;
opts.q = 2;
opts.const1 = (opts.p * opts.q) / (opts.p + opts.q);
opts.const2 = opts.p / (opts.p + opts.q);
opts.learning_rate_A = 1;
opts.learning_rate_B = 1;
opts.learning_rate_C = 1;
opts.lambda_A = 1e-3;
opts.lambda_B = 1e-3;
opts.lambda_C = 1e-3;

% main function to calculate W
[W_new,loss] = MLPC(X_train,X_train_process,Y_train,opts);
toc;
% display the convergence result
fig = figure;
plot(loss,'linewidth',3);
set(gca, 'fontsize',40);
xlabel('Number of iterations','fontsize',44);
ylabel('Objective value','fontsize',44);
set(gcf, 'Position',  [100, 100, 900, 700])
grid on
saveas(fig,'./plot/Convergence yeast.fig');
saveas(fig,'./plot/Convergence yeast.png');