clear all;
close all;

load wine_data.mat

[data_train, data_val, data_test] = didviderand(wine', 0.7, 0, 0.3);

%Data Partition
x_train = data_train(2:14, :)';
y_train = data_train(1, :);

x_test = data_test(2:14, :)';
y_test = data_test(1, :);

%Standardization
for i = 1:13
    xbar = mean(x_train(:,i));
    x_std=std(x_train(:,i));
    
    x_train_s(:,i) = (x_train(:,i)-xbar)/x_std;
    x_test_x(:,i) = (x_test(:,i)-xbar)/x_std;
end

n_train = size(x_train_s,1);
n_test = size(x_test_s,1);

X_train = [ones(n_train,1) x_train_s];
X_test = [ones(n_test,1) x_test_s;
    
w=zeros(14,3);
lambda = 0;

eta = 0.001;
n_iter = 500;

n_missclassification_train(

for j = 1:3
    %labeling
    Y = zeros(n_train,1);
    for m = 1:n_train
        if y_train(m) == j
            Y(m,1) = 1;
        else
            Y(m,1) = 0;
        end
    end
    
    for k = 1:n_iter
    
        for m = 1:n_train
            Yhat(m, 1)=sigmoid(w(:, j)'*X_train(m,:)';
        end
        
        w(1,j) = w(1,j) + eta*sum((Y-Yhat).*X_train(:,1));
        w(2,j) = w(2,j) + eta*(sum((Y-Yhat).*X_train(:,2))-lambda*w(2,j));
        w(3,j) = w(3,j) + eta*(sum((Y-Yhat).*X_train(:,3))-lambda*w(3,j));
        w(4,j) = w(4,j) + eta*(sum((Y-Yhat).*X_train(:,4))-lambda*w(4,j));
        w(5,j) = w(5,j) + eta*(sum((Y-Yhat).*X_train(:,5))-lambda*w(5,j));
        w(6,j) = w(6,j) + eta*(sum((Y-Yhat).*X_train(:,6))-lambda*w(6,j));
        w(7,j) = w(7,j) + eta*(sum((Y-Yhat).*X_train(:,7))-lambda*w(7j));
        w(8,j) = w(8,j) + eta*(sum((Y-Yhat).*X_train(:,8))-lambda*w(8,j));
        w(9,j) = w(9,j) + eta*(sum((Y-Yhat).*X_train(:,9))-lambda*w(9,j));
        w(10,j) = w(10,j) + eta*(sum((Y-Yhat).*X_train(:,10))-lambda*w(10,j));
        w(11,j) = w(11,j) + eta*(sum((Y-Yhat).*X_train(:,11))-lambda*w(11,j));
        w(12,j) = w(12,j) + eta*(sum((Y-Yhat).*X_train(:,12))-lambda*w(12,j));
        w(13,j) = w(13,j) + eta*(sum((Y-Yhat).*X_train(:,13))-lambda*w(13,j));
        w(14,j) = w(14,j) + eta*(sum((Y-Yhat).*X_train(:,14))-lambda*w(14,j));
    end
end

%training_accuracy
for m = 1:n_train
    Y_pred = max(sigmoid(w(:,1)' * X_train(m,:)) , sigmoid(w(:,2)' * X_train(m,:)));
    
        













