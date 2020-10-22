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
nSamples = 50; % Set the number of points sampled from given groundturths
AllSamples = cell(2,1);
for i = 1:2
    AllSamples{i} = mvnrnd(MeCentroid(i,:),SdCentroid(i,:),nSamples);
end
%% -- Visualize the training points -- %%
figure(H);
cla;
hold on;
plot(AllSamples{1}(:,1), AllSamples{1}(:,2), '^g', 'MarkerFaceColor', 'g');
plot(MeCentroid(1,1), MeCentroid(1,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(AllSamples{2}(:,1), AllSamples{2}(:,2), 'sb', 'MarkerFaceColor', 'b');
plot(MeCentroid(2,1), MeCentroid(2,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold off;
axis equal;
%% -- Between-Class scatter matrix -- %%
mu1 = mean(AllSamples{1});
mu2 = mean(AllSamples{2});
Sb = (mu1-mu2)'*(mu1-mu2); 
%% -- Within-Class scatter matrix -- %%
diff1 = AllSamples{1} - repmat(mu1,[nSamples,1]);
diff2 = AllSamples{2} - repmat(mu2,[nSamples,1]);
sigma1 = diff1'*diff1;
sigma2 = diff2'*diff2;
Sw = sigma1+sigma2; 
%% -- To maximize the generalized Rayleigh quotient -- %%
% To maximize J = (w'*Sb*w)/(w'Sw*w) is equivalent to mimimize E = -(w'*Sb*w), s.t. w'*Sw*w=1
% This can be solved by using Lagrange multiplier with SVD decomposition (stable solution)
[U,S,V] = svd(Sw); 
SwInv = V'*(S\U');
w = SwInv*(mu1-mu2)';
w = w*sign(w(1))/sqrt(sum(w.^2)); % normalize to a unit vector
%% -- compute the projections and visualize w -- %%
ProjectedLength{1} = sum(AllSamples{1} .* repmat(w',[nSamples,1]),2);
ProjectedLength{2} = sum(AllSamples{2} .* repmat(w',[nSamples,1]),2);
ProjectSamples{1} = repmat(ProjectedLength{1},[1,2]).* repmat(w',[nSamples,1]);
ProjectSamples{2} = repmat(ProjectedLength{2},[1,2]).* repmat(w',[nSamples,1]);
maxLen = max(max(ProjectedLength{1}),max(ProjectedLength{2})) + 2;
minLen = min(min(ProjectedLength{1}),min(ProjectedLength{2})) - 2;
figure(H);
xlim auto
ylim auto
hold on;
plot([minLen*w(1),maxLen*w(1)], [minLen*w(2),maxLen*w(2)],'r-','LineWidth', 3.0);
plot(ProjectSamples{1}(:,1), ProjectSamples{1}(:,2), '+g', 'MarkerFaceColor', 'g');
plot(ProjectSamples{2}(:,1), ProjectSamples{2}(:,2), '*b', 'MarkerFaceColor', 'b');
for i = 1:nSamples
    plot([AllSamples{1}(i,1),ProjectSamples{1}(i,1)],[AllSamples{1}(i,2),ProjectSamples{1}(i,2)], 'g--');
    plot([AllSamples{2}(i,1),ProjectSamples{2}(i,1)],[AllSamples{2}(i,2),ProjectSamples{2}(i,2)], 'b--');
end
hold off;
