%% matlab study
%% in Introductory Applied Machine Learning class
%% for machine learning

clear all;
close all;

load iris_data.mat

figure(1)
hold on
plot(x(1:50, 1), x(1:50, 2), 'ro')
plot(x(51:100, 1), x(51:100, 2), 'bx')