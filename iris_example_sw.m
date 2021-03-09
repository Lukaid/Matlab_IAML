clear all
close all

load iris_data.mat

figure(1)
hold on
plot(x(1:50,1), x(1:50,2), 'ro')
plot(x(51:100,1), x(51:100,2), 'cx')

legend('setosa','versicolor')
xlabel('sepal length')
ylabel('sepal width')
%test
x_test=[6 3.5;4 4;5.5 3.5;6.5 3.5;5 4;4.7 2.5;5.5 4;5 3.4;6 2.5;5.3 3];
for i=1:length(x_test)
    y_test(i)=seongwoo_classifier(x_test(i, 1), x_test(i, 2));
end

figure(1)
for i=1:length(x_test)
    if y_test(i) == 0
        plot(x_test(i,1),x_test(i,2),'bo')
    else
        plot(x_test(i,1),x_test(i,2),'bx')
    end
end
