clear all, close all

% training data and test data
data = load("cw1a.mat");
x = data.x;
y = data.y;
z = linspace(-3.2, 3.2, 101)';

% mean, covariance and likelihood functions
meanfunc = []; hyp.mean = [];
covfunc = @covPeriodic; p = 0; l = 1; sf = 0; hyp.cov = [p, l, sf];
likfunc = @likGauss; sn = 0; hyp.lik = sn;

% optimization
hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% negative log-likelihood
nlml = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
nlml_opt = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% posterior mean and covariance
[m, s] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, z);

% plot intervals with upper and lower bounds
f = [m+2*sqrt(s); flipdim(m-2*sqrt(s),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+')
title(sprintf('GP Regression with Optimized Hyperparameters\nPeriod (p) = %.3f, Length Scale (l) = %.3f, Signal Variance (sf) = %.3f, Noise Variance (sn) = %.3f' ...
    , exp(hyp_opt.cov(1)), exp(hyp_opt.cov(2)), exp(hyp_opt.cov(3)), exp(hyp.lik)));