clear all;
close all;

load iris_data.mat
figure(1)
hold on
plot(x(1:50, 1), x(1:50, 2), 'ro')
plot(x(51:100, 1), x(51:100, 2), 'bx')

for i=1:100
    y_test(i) = seongwoo_classifier(x(i, 1), x(i, 2))
end

figure(2)
hold on
for i = 1:100
    if y_test(i) == 0
        plot(x(i, 1), x(i, 2), 'go')
    else
        plot(x(i, 1), x(i, 2), 'yx')
    end
end

x_ex = [4.5, 7]
y_ex = 2/3 * x_ex - 1/2
plot(x_ex, y_ex)