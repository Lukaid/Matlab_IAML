clear all
close all

%load iris_data
load iris_data.mat

X=[x(1:100,1) x(1:100,2)];
Y=y(1,1:100)';

figure(1)
hold on
plot(X(1:50,1),X(1:50,2),'ro')
plot(X(51:100,1),X(51:100,2),'bx')

xlabel('Sepal length')
ylabel('Sepal width')

%Perceptron
n_iter=1000; %# of epoch
eta=0.1; %learning rate
w=[3;3;3]; %initial value for w

for i=1:n_iter %# of epoch
    for j=1:length(X)
        
        %prediction
        Yhat(j,1)=step_f(w'*[1;X(j,:)']);
        
        %update
        w=w+eta*(Y(j,1)-Yhat(j,1))*[1;X(j,:)'];
    end    
end

x1=4:0.1:7;
for i=1:length(x1)
    x2(i)=-w(1)/w(3)-w(2)/w(3)*x1(i);
end

figure(1)
plot(x1,x2)