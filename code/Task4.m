clear all; close all;

% define the Covariance Function
covFunc = {@covProd, {@covPeriodic, @covSEiso}}; hyp.cov = [-0.5, 0, 0, 2, 0];

% covariance matrix
x = linspace(-5, 5, 200)'; 
K = feval(covFunc{:}, hyp.cov, x, x);

% Cholesky decomposition: K = LL^T
K_stable = K + 1e-6 * eye(length(x)); 
L = chol(K_stable, 'lower');

% sampling random functions
n_samples = 20; y = L * randn(length(x), n_samples);

figure;
hold on;
for i = 1:n_samples
    plot(x, y(:, i));
end
title(sprintf('Samples from Gaussian Process, with number of samples = %.f', n_samples)); xlabel('x'); ylabel('f(x)');
hold off;
