clear all; close all;

data = load("cw1e.mat"); x = data.x; y = data.y;

% model 1: GP with covSEard covariance
covfunc1 = @covSEard; hyp1.cov = ones(3,1); hyp1.lik = 0.1;   

% model 2: GP with {@covSum, {@covSEard, @covSEard}} covariance
covfunc2 = {@covSum, {@covSEard, @covSEard}}; hyp2.cov = 0.1 * randn(6,1); hyp2.lik = 0.1;        

% global Likelihood function
likfunc = @likGauss;

% optimization
hyp1_optimized = minimize(hyp1, @gp, -100, @infGaussLik, [], covfunc1, likfunc, x, y);
hyp2_optimized = minimize(hyp2, @gp, -100, @infGaussLik, [], covfunc2, likfunc, x, y);

% log marginal likelihoods
nlml1 = gp(hyp1_optimized, @infGaussLik, [], covfunc1, [], x, y);
nlml2 = gp(hyp2_optimized, @infGaussLik, [], covfunc2, [], x, y);

% Create a grid for prediction
[x1Grid, x2Grid] = meshgrid(linspace(min(x(:,1)), max(x(:,1)), 11), ...
                            linspace(min(x(:,2)), max(x(:,2)), 11));
xTest = [x1Grid(:), x2Grid(:)];

% Predictions for both models
[mu1, ~] = gp(hyp1_optimized, @infGaussLik, [], covfunc1, [], x, y, xTest);
[mu2, ~] = gp(hyp2_optimized, @infGaussLik, [], covfunc2, [], x, y, xTest);

% Reshape for plotting
mu1_grid = reshape(mu1, size(x1Grid)); mu2_grid = reshape(mu2, size(x1Grid));

% plot models
figure;
mesh(x1Grid, x2Grid, mu1_grid);
hold on;
scatter3(x(:,1), x(:,2), y, 'r', 'filled');  % Overlay data points
hold off;
title(sprintf('GP with covSEard\nlog marginal likelihood = %.3f', nlml1));

figure;
mesh(x1Grid, x2Grid, mu2_grid);
hold on;
scatter3(x(:,1), x(:,2), y, 'r', 'filled');  % Overlay data points
hold off;
title(sprintf('GP with {@covSum, {@covSEard, @covSEard}}\nlog marginal likelihood = %.3f', nlml2));

% Calculate residuals for both models
residuals1_squared = (y - mu1).^2;
residuals2_squared = (y - mu2).^2;

% Reshape the residuals for plotting
residuals1_grid = reshape(residuals1_squared, [11, 11]);
residuals2_grid = reshape(residuals2_squared, [11, 11]);

% 3D Bar plot for residuals of model 1
figure;
bar3(residuals1_grid);
title('3D Bar Plot of Squared Residuals for GP with covSEard');
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Residuals');

% 3D Bar plot for residuals of model 2
figure;
bar3(residuals2_grid);
title('3D Bar Plot of Squared Residuals for GP with {@covSum, {@covSEard, @covSEard}}');
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Residual Squared');