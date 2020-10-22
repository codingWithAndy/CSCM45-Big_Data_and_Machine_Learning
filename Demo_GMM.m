%% -- Clean up the workspace variables, and close all figures -- %%
clear
close all
clc
%% -- Sample the training points from given centroid and standard deviation -- %%
MeCentroid = zeros([2,2]);
H = figure;
xlim([0,15])
ylim([0,15])
hold on
for c = 1:2
    [MeCentroid(c,1),MeCentroid(c,2)] = ginput(1);
    plot(MeCentroid(c,1), MeCentroid(c,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r');  
end
hold off
SdCentroid = unifrnd(1,5,[2,2]); % Uniformly generates standard deviation within the given boundary
nSamples = 100; % Set the number of points sampled from given groundturths
AllSamples = cell(2,1);
for i = 1:2
    AllSamples{i} = mvnrnd(MeCentroid(i,:),SdCentroid(i,:),nSamples);
end
%% -- Visualize the training points -- %%
figure(H);
cla;
hold on;
plot(AllSamples{1}(:,1), AllSamples{1}(:,2), 'dg', 'MarkerFaceColor', 'g');
plot(MeCentroid(1,1), MeCentroid(1,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(AllSamples{2}(:,1), AllSamples{2}(:,2), 'sb', 'MarkerFaceColor', 'b');
plot(MeCentroid(2,1), MeCentroid(2,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold off;
axis equal;
%% -- GMM Fitting -- %%
warning('off','stats:gmdistribution:FailedToConverge')
Pts = cell2mat(AllSamples);
for r = 1:20
    options = statset('Display','off','MaxIter',1);
    if r == 1
        gm = fitgmdist(Pts,2,'Options',options);
        Likelihood = gm.NegativeLogLikelihood;
    else
        % Initialize current state with the results from previous iteration
        S = struct('mu', gm.mu, 'Sigma', gm.Sigma, 'ComponentProportion', gm.ComponentProportion); 
        gm = fitgmdist(Pts,2,'Options',options,'Start',S);
        % Check convergence via compute the difference of log likelihood between two iterations  
        if abs(gm.NegativeLogLikelihood-Likelihood) >= 0.1 
            Likelihood = gm.NegativeLogLikelihood;
        else
            disp(['Iteration ' num2str(r-1) ' Log Likelihood: -' num2str(Likelihood) '; Converged!']);
            break;
        end
    end
    disp(['Iteration ' num2str(r) ' Log Likelihood: -' num2str(Likelihood)]);
    % Visualize the PDF
    gmPDF = @(x,y)pdf(gm,[x y]);
    figure(H);
    cla;
    hold on
    plot(AllSamples{1}(:,1), AllSamples{1}(:,2), 'dg', 'MarkerFaceColor', 'g');
    plot(AllSamples{2}(:,1), AllSamples{2}(:,2), 'sb', 'MarkerFaceColor', 'b');
    ezcontour(gmPDF,[0 15],[0 15]);
    hold off
    pause(1);
end
%%
