clear all;
close all;

%load iris_data
load iris_data.mat

X = [x(1:100, 1), x(1:100, 3)]

for i = 1:100;
    if y(i) == 0
        Y(i, 1) = 1;
    else y(i) == 1
        Y(i, 1) = -1;
    end
end
%Y = [y(1:100)]' % 이렇게 ' 요거 찍으면 전치

figure(1)
hold on
plot(X(1:50, 1), X(1:50, 2), 'ro')
plot(X(51:100, 1), X(51:100, 2), 'bx')

xlabel('Sepal lengnth')
ylabel('Petal lengnth')
legend('setosa','versicolor')

%Perceptron
n_iter =  10;
eta = 0.1; % 이게 뭐지? runnig rate?

w = [0;0;0] % initial guess


for i = 1:n_iter
    
    % prediction
    for j = 1:length(X)
        if w'*[1;X(j, :)'] >= 1
            Yhat(j, 1) = 1;
        else 
            Yhat(j, 1) = -1;
        end   
    end
    
    % update
    for j = 1:length(X)
        w = w + eta*(Y(j, 1) - Yhat(j, 1))*[1;X(j, :)']
    end 
end

x1 = 4:0.1:7;

for i = 1:length(x1)
    x2(i) = -w(1)/w(3) -w(2)/w(3)*x1(i);
end

figure(1)
plot(x1, x2)



