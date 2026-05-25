# Gaussian Process Regression

An implementation of Gaussian Process regression in MATLAB, covering covariance function design, hyperparameter optimisation, kernel composition, and Bayesian model comparison.

Built on Carl Edward Rasmussen's [4F13 course](https://mlg.eng.cam.ac.uk/teaching/4f13/2425/) at the University of Cambridge. Awarded First Class.

---

## Overview

Gaussian Processes offer a principled, fully Bayesian approach to regression — instead of fitting a single function to data, a GP maintains a distribution over functions and updates it as evidence arrives. This repo works through that idea from the ground up: starting with a basic SE kernel, examining how hyperparameter choices shape the posterior, moving to periodic and composite kernels, and finishing with a formal model comparison on 2D data.

---

## Contents

| File | Description |
|---|---|
| `Task1.m` | GP regression with SE isotropic covariance; hyperparameter optimisation via marginal likelihood |
| `Task2.m` | Systematic grid search over hyperparameter initialisations to study sensitivity and local optima |
| `Task3.m` | GP regression with a periodic covariance function |
| `Task3_hyp_finder.m` | Extended 4-dimensional grid search for periodic hyperparameters; exports ranked results to CSV |
| `Task4.m` | Sampling from a GP prior using a product kernel (`covPeriodic × covSEiso`) via Cholesky decomposition |
| `Task5.m` | Model comparison on 2D data: `covSEard` vs. sum of two ARD kernels, with residual analysis |
| `cw1a.mat` | 1D dataset (Tasks 1–3) |
| `cw1e.mat` | 2D dataset (Tasks 4–5) |
| `coursework1_questions.pdf` | Original problem sheet |
| `coursework1_submission.pdf` | Written report |

---

## Key ideas explored

**Hyperparameter sensitivity (Tasks 2 & 3)**
Marginal likelihood optimisation is non-convex — the optimised solution depends on where you start. `Task2.m` and `Task3_hyp_finder.m` run structured grid searches over initialisations to map out how sensitive the final fit is to starting conditions, and to avoid poor local optima. `Task3_hyp_finder.m` logs every trial to CSV for post-hoc analysis.

**Kernel composition (Task 4)**
Rather than picking a single covariance function, a product kernel `covPeriodic × covSEiso` combines periodicity with a decaying envelope. Sampling directly from this prior via Cholesky decomposition makes the inductive bias of the kernel concrete and interpretable.

**Bayesian model comparison (Task 5)**
Two models are compared on 2D data using the log marginal likelihood as a principled score. Beyond the scalar likelihood, squared residuals are visualised as 3D bar charts across the input space to identify where each model's inductive bias breaks down.

---

## Requirements

- MATLAB (tested on R2023b)
- [GPML Toolbox](http://gaussianprocess.org/gpml/code/matlab/doc/) on the MATLAB path

## Running

```matlab
run('Task1.m')
```
