% function to evaluate MLPC with ACC, AUC, Macro F1 and Micro F1 
% Input : training data (double): X_train
%         testing data (double): X_test, Y_test
%         core tensor for training data: W_new
% Output : four metrics: ACC, AUC, Macro F1 and Micro F1
%          Predicted Labels (double): Y_pred
%          core tensor for testing data: W_hat
function [acc,ham,auc,macro_F1,micro_F1,Y_pred,W_hat] = evalMLPC(X_test, Y_test, X_train, W_new)
    % problem size
    N = size(Y_test,1);
    L = size(Y_test,2);
    D = size(X_train,2);
    Y_pred = zeros(N,L);

    k = 5;
    % weber solver 
    [neighbors, distances] =  knnsearch(X_train, X_test, 'K', k, 'IncludeTies',true);
    neighborW = cell(N,k);
    neighborWeight = cell(N,k);
    for i = 1:N
        for j = 1:k
            neighborW{i,j} = double(W_new(:,neighbors{i}(j)+1,:));
            neighborWeight{i,j} = exp(-(0.01).*distances{i}(j));
        end
    end

    W_hat = Weber_Solver(neighborW,neighborWeight,N,D,L);
    temp = zeros(N,L);
    % simple thresholding with 0.5
    for n = 1:N
        temp(n,:) = double(W_hat(:,n,:) + W_new(:,1,:)) * (X_test(n,:))';
        for l = 1:L
            if temp(n,l) > 0.5
                Y_pred(n,l) = 1;
            end
        end
    end

    acc = Accuracy(Y_pred',Y_test');
    ham = Hamming_score(Y_pred',Y_test');
    auc = average_auroc(temp',Y_test');
    macro_F1 = Macro_F1(Y_pred',Y_test');
    micro_F1 = Micro_F1(Y_pred',Y_test');   
end
%% private function for weber problem
function W_hat = Weber_Solver(neighborW,neighborWeight,N,D,L)
    W_hat = tenzeros([L,N,D]);
    for iter = 1:100
        temp1 = zeros(L,D);
        temp2 = 0;
        for i = 1:N
            for j = 1:5
                temp1 = temp1 + neighborWeight{i,j} * neighborW{i,j} ./ (norm(double(W_hat(:,i,:) - neighborW{i,j}),'fro') + eps);
                temp2 = temp2 + neighborWeight{i,j} ./ (norm(double(W_hat(:,i,:) - neighborW{i,j}),'fro') + eps);
                W_hat(:,i,:) = temp1 ./ temp2;
            end 
        end
    end
end