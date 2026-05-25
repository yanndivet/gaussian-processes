clear all; close all;

% training data and test data
data = load("cw1a.mat");
x = data.x;
y = data.y;
z = linspace(-3.2, 3.2, 101)';

% mean, covariance, and likelihood functions
meanfunc = [];
hyp.mean = [];
covfunc = @covPeriodic;
likfunc = @likGauss;

% Define the ranges for p, l, sf, and sn values you want to iterate through
p = [-2, -1];
l = [-2, 1.5, 2];
sf = [-0.5, -0.1];
sn = [-3, -2.5];

results = [];

% Loop through hyperparameter values
for i = 1:numel(p)
    for j = 1:numel(l)
        for k = 1:numel(sf)
            for n = 1:numel(sn)
                hyp.cov = [p(i), l(j), sf(k)];
                hyp.lik = sn(n);

                % Optimization
                hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

                % Negative log-likelihood
                nlml = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
                nlml_opt = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

                % Store results
                result = struct('p', p(i), 'l', l(j), 'sf', sf(k), 'sn', sn(n), 'nlml', nlml, 'nlml_opt', nlml_opt);
                results = [results; result];

                figure;

                % Posterior mean and covariance
                [m, s] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, z);

                % Plot intervals with upper and lower bounds
                f = [m + 2 * sqrt(s); flipdim(m - 2 * sqrt(s), 1)];
                fill([z; flipdim(z, 1)], f, [7 7 7] / 8)
                hold on
                plot(z, m)
                plot(x, y, '+')
                title(sprintf('GP Regression with Optimized Hyperparameters\nPeriod (p) = %.3f, Length Scale (l) = %.3f, Signal Variance (sf) = %.3f, Noise Variance (sn) = %.3f\nAnd optimized NLML: %.3f' ...
                    , exp(hyp_opt.cov(1)), exp(hyp_opt.cov(2)), exp(hyp_opt.cov(3)), exp(hyp_opt.lik)), nlml_opt);
                xlabel('Input (z)')
                ylabel('Output (m)')
                hold off
            end
        end
    end
end

table = struct2table(results);
writetable(table, 'hyperparameter_results_periodic.csv');
