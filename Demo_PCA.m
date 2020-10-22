%% -- Clean up the workspace variables, and close all figures -- %%
clear
close all
clc
%% Generate random data point given ground turth
mu = [3,4]; % Ground turth centre
% sigma = rndSPDMatrix(2); % Ground turth covariance (symmetric positive definite)
sigma = [1,1.5;1.5,3];
% Example [1,1.5;1.5,3];
Samples = mvnrnd(mu, sigma, 250); % Generate sample points
% visualize the data points
h = figure;
plot(Samples(:,1), Samples(:,2), 'g+');
%% Apply PCA to the data points
% compute the data mean
ptsMean = mean(Samples);
% substract the data mean for each data point
nSamples = (Samples - repmat(ptsMean,[size(Samples,1),1]))/sqrt(size(Samples,1)-1);
% SVD decomposition
[U,S,V] = svd(nSamples,0);
% Compute Eigen value
Evalues = diag(S).^2;
% Compute Eigen vector and make the sign of major direction positive (align to X+ direction)
Evectors = bsxfun(@times,V,sign(V(1,:)));
%%
figure(h);
hold on;
plot(ptsMean(1), ptsMean(2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
quiver(ptsMean(1), ptsMean(2), sqrt(Evalues(1))*Evectors(1,1), sqrt(Evalues(1))*Evectors(1,2), 'r-', 'LineWidth', 3.0);
quiver(ptsMean(1), ptsMean(2), sqrt(Evalues(2))*Evectors(2,1), sqrt(Evalues(2))*Evectors(2,2), 'b-', 'LineWidth', 3.0);
hold off;
axis equal
%%
