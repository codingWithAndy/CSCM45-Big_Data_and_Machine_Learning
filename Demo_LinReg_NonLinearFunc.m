%% -- Clean up the workspace variables, and close all figures -- %%
clear
close all
clc
%% 2D regression with linear function
% Generate the 2D data with unknow function
% We assume that the function is y = w0 + w1*x where w0 = 1.2; w1=3.25;
% However, we normally end up with the observation with some noisy;
X = 0.1:0.1:1;
Y_True = exp(1.4*X);
Y_Observation = Y_True + 0.5*normrnd(0,0.5,1,numel(X));

disp('Display the true data and the observations');
% Plot the true function, and the observations
h1 = figure;
hold on;
plot(-0.2:0.1:1.2, exp(1.4*[-0.2:0.1:1.2]), '-g', 'LineWidth', 2); % plot true function in green
plot(X, Y_Observation, 'ob', 'MarkerSize', 10, 'LineWidth', 2); % plot observation of Y in blue
plot([X;X], [Y_True;Y_Observation], '-c', 'LineWidth', 2); % plot the distance between the observations and the true values
xlim([-0.2, 1.2]);
ylim([-1, 5]);
hold off;
%% Use linear kernel to fit
pause();
disp('Fitting the linear function with gradient descent');
% Fit a linear function to the data
% let initial the w0 and w1 with some random number
w0 = 2*rand(); %make the value close to the true w0
w1 = 6*rand(); %make the value close to the true w1
% learning speed control parameter
alpha = 0.5; % change the value of learning speed to see the difference
% gradient descent, you can have a much better stopping criterion
disp('->Gradient Descent');
h2 = figure;
for i = 1:200 % change the number of search step to see the difference
    Y_Fit = w0 + w1*X; % the value of the model with the parameters w0 and w1
    Y_Dif = Y_Fit - Y_Observation; % the difference between the observations and the value given by the model
    PD_MSE_W0 = sum(Y_Dif.*1); % the partial dirivative of mean squares of error with respect to w0
    w0 = w0 - alpha*PD_MSE_W0/numel(Y_Observation); % update the w0
    
    PD_MSE_W1 = sum(Y_Dif.*X); % the partial dirivative of mean squares of error with respect to w1 
    w1 = w1 - alpha*PD_MSE_W1/numel(Y_Observation); % update the w1
    
    % display the message and plot the graph every 10 runs
    MSE = mean(((w0 + w1*X) - Y_Observation).^2);
    if mod(i,10) == 0
        disp(['->->Itor: ' num2str(i) ' Update w0 = ' num2str(w0) ' w1 = ' num2str(w1) ' MSError = ' num2str(MSE)]);
        figure(h2);
        cla;
        hold on;
        plot(-0.2:0.1:1.2, exp(1.4*[-0.2:0.1:1.2]), '-g', 'LineWidth', 2); % plot true function in green
        plot(-0.2:0.1:1.2, w0+w1*(-0.2:0.1:1.2), '-r', 'LineWidth', 2); % plot the fitted function in green
        plot(X, Y_Observation, 'ob', 'MarkerSize', 10, 'LineWidth', 2); % plot observation of Y in blue
        plot([X;X], [(w0 + w1*X);Y_Observation], '-c', 'LineWidth', 2); % plot the distance between the observations and the fitted values
        xlim([-0.2, 1.2]);
        ylim([-1, 5]);
        hold off;
        pause(1);
    end
end
disp(['-->Model: w0 = ' num2str(w0) ' w1= ' num2str(w1) ' MSError = ' num2str(MSE)]);
%% Use nonlinear kernel model to fit
pause();
disp('Fitting the linear function with gradient descent');
% Fit a linear function to the data
% let initial the w0 and w1 with some random number
w0 = 2*rand(); 
w1 = 2*rand(); 
w2 = 2*rand(); 
% learning speed control parameter
alpha = 0.5; % change the value of learning speed to see the difference
% gradient descent, you can have a much better stopping criterion
disp('->Gradient Descent');
h2 = figure;
for i = 1:200 % change the number of search step to see the difference
    Y_Fit = w0 + w1*X + w2*X.^2; % the value of the model with the parameters w0 and w1
    Y_Dif = Y_Fit - Y_Observation; % the difference between the observations and the value given by the model
    % Update w0
    PD_MSE_W0 = sum(Y_Dif.*1); % the partial dirivative of mean squares of error with respect to w0
    w0 = w0 - alpha*PD_MSE_W0/numel(Y_Observation); % update the w0
    % Update w1
    PD_MSE_W1 = sum(Y_Dif.*X); % the partial dirivative of mean squares of error with respect to w1 
    w1 = w1 - alpha*PD_MSE_W1/numel(Y_Observation); % update the w1
    % Update w2
    PD_MSE_W2 = sum(Y_Dif.*X.^2); % the partial dirivative of mean squares of error with respect to w1 
    w2 = w2 - alpha*PD_MSE_W2/numel(Y_Observation); % update the w1
    
    % display the message and plot the graph every 10 runs
    MSE = mean(((w0 + w1*X + w2*X.^2) - Y_Observation).^2);
    if mod(i,10) == 0
        disp(['->->Itor: ' num2str(i) ' Update w0 = ' num2str(w0) ' w1 = ' num2str(w1) ' w2= ' num2str(w2) ' MSError = ' num2str(MSE)]);
        figure(h2);
        cla;
        hold on;
        plot(-0.2:0.1:1.2, exp(1.4*[-0.2:0.1:1.2]), '-g', 'LineWidth', 2); % plot true function in green
        plot(-0.2:0.1:1.2, w0+w1*(-0.2:0.1:1.2)+w2*(-0.2:0.1:1.2).^2, '-r', 'LineWidth', 2); % plot the fitted function in green
        plot(X, Y_Observation, 'ob', 'MarkerSize', 10, 'LineWidth', 2); % plot observation of Y in blue
        plot([X;X], [(w0 + w1*X + w2*X.^2);Y_Observation], '-c', 'LineWidth', 2); % plot the distance between the observations and the fitted values
        xlim([-0.2, 1.2]);
        ylim([-1, 5]);
        hold off;
        pause(1);
    end
end
disp(['-->Model: w0 = ' num2str(w0) ' w1= ' num2str(w1) ' w2= ' num2str(w2) ' MSError = ' num2str(MSE)]);
