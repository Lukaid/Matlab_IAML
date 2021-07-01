clear all;
close all;

load wine.data

[data_train, data_val, data_test] = dividerand(wine', 0.7, 0, 0.3);

%Data Partition
x_train = data_train(2:14, :)';
y_train = data_train(1, :)';

x_test = data_test(2:14, :)';
y_test = data_test(1, :)';

%Standardization
for i = 1:13
    xbar = mean(x_train(:,i));
    x_std=std(x_train(:,i));
    
    x_train_s(:,i) = (x_train(:,i)-xbar)/x_std;
    x_test_s(:,i) = (x_test(:,i)-xbar)/x_std;
end

n_train = size(x_train_s,1); %1은 행의 개수, 2는 열의 개수
n_test = size(x_test_s,1);

X_train = [ones(n_train,1) x_train_s]; %ones(n_train,1) n_train만큼 칼럼은 하나만
X_test = [ones(n_test,1) x_test_s];
    
w=zeros(14,3); % one vs all (class가 3개니까)
lambda = 0:10:300;

eta = 0.001;
n_iter = 500;

n_missclassification_train = zeros(length(lambda), 1);
n_missclassification_test = zeros(length(lambda), 1);

for i = 1:length(lambda)

    %training
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
                Yhat(m, 1)=sigmoid(w(:, j)'*X_train(m,:)');
            end

            w(1,j) = w(1,j) + eta*sum((Y-Yhat).*X_train(:,1));
            w(2,j) = w(2,j) + eta*(sum((Y-Yhat).*X_train(:,2))-lambda(i)*w(2,j));
            w(3,j) = w(3,j) + eta*(sum((Y-Yhat).*X_train(:,3))-lambda(i)*w(3,j));
            w(4,j) = w(4,j) + eta*(sum((Y-Yhat).*X_train(:,4))-lambda(i)*w(4,j));
            w(5,j) = w(5,j) + eta*(sum((Y-Yhat).*X_train(:,5))-lambda(i)*w(5,j));
            w(6,j) = w(6,j) + eta*(sum((Y-Yhat).*X_train(:,6))-lambda(i)*w(6,j));
            w(7,j) = w(7,j) + eta*(sum((Y-Yhat).*X_train(:,7))-lambda(i)*w(7,j));
            w(8,j) = w(8,j) + eta*(sum((Y-Yhat).*X_train(:,8))-lambda(i)*w(8,j));
            w(9,j) = w(9,j) + eta*(sum((Y-Yhat).*X_train(:,9))-lambda(i)*w(9,j));
            w(10,j) = w(10,j) + eta*(sum((Y-Yhat).*X_train(:,10))-lambda(i)*w(10,j));
            w(11,j) = w(11,j) + eta*(sum((Y-Yhat).*X_train(:,11))-lambda(i)*w(11,j));
            w(12,j) = w(12,j) + eta*(sum((Y-Yhat).*X_train(:,12))-lambda(i)*w(12,j));
            w(13,j) = w(13,j) + eta*(sum((Y-Yhat).*X_train(:,13))-lambda(i)*w(13,j));
            w(14,j) = w(14,j) + eta*(sum((Y-Yhat).*X_train(:,14))-lambda(i)*w(14,j));
        end
    end

    %for ploting
    mean_w(:, i)=(w(2:14, 1) + w(2:14, 2) + w(2:14, 3)) / 3;



    %training_accuracy
    for m = 1:n_train
        [temp, Y_pred] = max([sigmoid(w(:,1)' * X_train(m,:)'), sigmoid(w(:,2)' * X_train(m,:)'), sigmoid(w(:,3)' * X_train(m,:)')]);

        if (y_train(m,1)-Y_pred) ~= 0
            n_missclassification_train(i,1) = n_missclassification_train(i,1) + 1;
        end
    end
    train_accuracy(i,1) = 1-n_missclassification_train(i,1)/n_train;

    %test_accuracy
    for m = 1:n_test
        [temp, Y_pred] = max([sigmoid(w(:,1)' * X_test(m,:)'), sigmoid(w(:,2)' * X_test(m,:)'), sigmoid(w(:,3)' * X_test(m,:)')]);

        if (y_test(m,1) - Y_pred) ~= 0
            n_missclassification_test(i,1) = n_missclassification_test(i,1) + 1;
        end
    end
    test_accuracy(i,1) = 1-n_missclassification_test(i,1)/n_test;
end

figure(1)
hold on
plot(lambda, mean_w(1,:))
plot(lambda, mean_w(2,:))
plot(lambda, mean_w(3,:))
plot(lambda, mean_w(4,:))
plot(lambda, mean_w(5,:))
plot(lambda, mean_w(6,:))
plot(lambda, mean_w(7,:))
plot(lambda, mean_w(8,:))
plot(lambda, mean_w(9,:))
plot(lambda, mean_w(10,:))
plot(lambda, mean_w(11,:))
plot(lambda, mean_w(12,:))
plot(lambda, mean_w(13,:))
legend('w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13')
xlabel('lambda')
ylabel('w')


figure(2)
hold on
plot(lambda, train_accuracy)
plot(lambda, test_accuracy, 'r--')
xlabel('lambda')
ylabel('accuracy')
legend('train', 'test')






