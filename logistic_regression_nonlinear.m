%Logistic regression
%nonlinear

clear all
close all

%data generation
% theta=-pi+2*pi*rand(100,1); %generate random -pi<= theta <= pi
% r1=rand(50,1);
% r2=1+rand(50,1);
% r=[r1;r2];
% x=[r1.*cos(theta(1:50,1)) r1.*sin(theta(1:50,1));
%    r2.*cos(theta(51:100,1)) r2.*sin(theta(51:100,1))]
load circle_data.mat

figure(1)
hold on
plot(x(1:50,1),x(1:50,2),'ro')
plot(x(51:100,1),x(51:100,2),'bx')

X=[ones(100,1) x x.^2];
Y=[zeros(50,1);ones(50,1)];

w=[0;0;0;0;0];
eta=0.0001;
n_iter=10000;

likelihood=ones(n_iter,1);
n_missclassification=zeros(n_iter,1);

for i=1:n_iter

    %predict
    
    for j=1:length(X)
        Yhat(j,1)=sigmoid(w'*X(j,:)');
        
        if (Y(j,1)-output_activation_logistic(Yhat(j,1))) ~= 0
            n_missclassification(i,1)=n_missclassification(i,1)+1;
        end
        
        likelihood(i,1)=likelihood(i,1)*Yhat(j,1)^(Y(j,1))*(1-Yhat(j,1))^(1-Y(j,1));
    end
    
    %update
    
    w(1)=w(1)+eta*sum((Y-Yhat).*X(:,1));
    w(2)=w(2)+eta*sum((Y-Yhat).*X(:,2));
    w(3)=w(3)+eta*sum((Y-Yhat).*X(:,3));
    w(4)=w(4)+eta*sum((Y-Yhat).*X(:,4));
    w(5)=w(5)+eta*sum((Y-Yhat).*X(:,5));
                   
end

%x1=-3:0.1:3;
%for i=1:length(x1)
%    x2(i)=-w(1)/w(3)-w(2)/w(3)*x1(i);
%end

x1=linspace(-3,3);
x2=linspace(-3,3);
[X1,X2]=meshgrid(x1,x2);
Z = w(1) + w(2)*X1 + w(3)*X2 + w(4)*X1.^2 + w(5)*X2.^2;

figure(1)
%plot(x1,x2)
contourf(X1,X2,Z,[0 0])
plot(x(1:50,1),x(1:50,2),'ro')
plot(x(51:100,1),x(51:100,2),'bx')


figure(2)
plot(1:n_iter, n_missclassification)

figure(3)
plot(1:n_iter, -log(likelihood))
