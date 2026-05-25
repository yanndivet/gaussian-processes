clear all; close all;
% Tasks (a), (b), (c) share the same algorithm structure
% Just replacing the covariance functions and hyper-parameters

% training data and test data
data = load("cw1a.mat");
x = data.x; y = data.y; z = linspace(-3.2, 3.2, 101)';

% initialization
meanfunc = []; covfunc = @covSEiso; likfunc = @likGauss;
hyp.mean = [];
ell = [-9.4, -9.3]; % different ell
sf = [-3, -2.5, -0.5, -0.1]; % different sf
sn = [0.05, 0.1, 0.5, 0.7, 1]; % different sn

% loop through hyperparameters
for i = 1:numel(ell)
    for j = 1:numel(sf)
        for k = 1:numel(sn)
            hyp.cov = [ell(i), sf(j)]; 
            hyp.lik = sn(k); 

            % optimization
            hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

            % negative log-likelihood
            nlml = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
            nlml_opt = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

            % posterior mean and covariance
            [m, s] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, z);

            % plot
            figure;
            f = [m + 2 * sqrt(s); flipdim(m - 2 * sqrt(s), 1)];
            fill([z; flipdim(z, 1)], f, [7 7 7] / 8)
            hold on
            plot(z, m)
            plot(x, y, '+')
            title(sprintf('GP Regression with Optimized Hyperparameters\nLength Scale (l) = %.3f, Signal Variance (sf) = %.3f, Noise Variance (sn) = %.3f\nAnd optimized NLML: %.3f' ...
                , exp(hyp_opt.cov(1)), exp(hyp_opt.cov(2)), exp(hyp_opt.lik)), nlml_opt);
            xlabel('Input (z)')
            ylabel('Output (m)')
        end
    end
end