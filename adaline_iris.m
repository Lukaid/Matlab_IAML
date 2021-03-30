clear all
close all

%load iris_data
load iris_data.mat

X=[x(1:100,1) x(1:100,2)];
for i = 1:100
    if y(i) == 0
        Y(i, 1) = 1;
    else y(i) == 1;
        Y(i, 1) = -1;
    end
end

figure(1)
hold on
plot(X(1:50,1),X(1:50,2),'ro')
plot(X(51:100,1),X(51:100,2),'bx')

xlabel('Sepal length')
ylabel('Sepal width')

%Adaptive linear neuron
n_iter=10000; %# of epoch
eta=0.0001; %learning rate
w=[3;3;3]; %initial value for w

sum_square_error = zeros(n_iter, 1);
for i=1:n_iter % # of epoch
    
    error_sum = zeros(3, 1);
    
    for j=1:length(X)
        
        %prediction
        Yhat(j,1)=w'*[1;X(j,:)']; % linear function을 통화하는 것이니 그대로
        
        error_sum = error_sum + (Y(j,1)-Yhat(j,1))*[1;X(j,:)']; % 바로 업데이트 하는것이 아닌 error를 다 모아서 업데이트
        
        sum_square_error(i, 1) = sum_square_error(i, 1) + (Y(j,1)-Yhat(j,1))^2;
        
    end
        %update
        w=w+eta*error_sum;  
end

x1=4:0.1:7;
for i=1:length(x1)
    x2(i)=-w(1)/w(3)-w(2)/w(3)*x1(i);
end

figure(1)
plot(x1,x2)

figure(2)
plot(sum_square_error)
sum_square_error(n_iter, 1)