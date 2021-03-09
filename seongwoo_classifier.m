function y = seongwoo_classifier(x_1, x_2)
%% point_1 : (4.5, 2.5)
%% point_2 : (5.4, 3.1)
%% gradient = 2/3
%% function : x_2 = 2/3 * x_1 - 1/2


if x_2 >= 2/3 * x_1 - 1/2
    y = 0
else
    y = 1    
end