%% funtion to preprocess X
function X_process = data_preprocess(X)
    [N,D] = size(X);
    temp = zeros(N,N*D);
    for i = 1:N
        temp(i,(i-1)*D+1:i*D) = X(i,:);
    end
    X_process = [X,sparse(temp)];
end

