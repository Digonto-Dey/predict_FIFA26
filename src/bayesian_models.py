import numpy as np
import pandas as pd
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

class BayesianMatchPredictor:
    def __init__(self):
        self.trace = None
        self.model = None
        
    def fit(self, X_train, y_train, draws=500, tune=500):
        if not PYMC_AVAILABLE:
            print("PyMC not available. Skipping Bayesian model fit.")
            return
            
        print(f"Building PyMC Bayesian Multinomial Logistic Regression model on {len(X_train)} samples...")
        
        # Simplify features for Bayesian model to keep it tractable
        # Standardize features
        features_to_use = ['rating_difference', 'home_form_pts_5', 'away_form_pts_5', 'is_neutral']
        X = X_train[features_to_use].copy()
        
        for col in features_to_use:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-9)
            
        X_mat = X.values
        y_val = y_train.values
        
        with pm.Model() as self.model:
            # Priors for the regression coefficients
            # Shape is (num_features, num_classes)
            alpha = pm.Normal('alpha', mu=0, sigma=1, shape=3)
            beta = pm.Normal('beta', mu=0, sigma=1, shape=(X_mat.shape[1], 3))
            
            # Linear model
            mu = alpha + pm.math.dot(X_mat, beta)
            
            # Softmax to get probabilities
            p = pm.math.softmax(mu, axis=1)
            
            # Likelihood
            y_obs = pm.Categorical('y_obs', p=p, observed=y_val)
            
            print("Starting MCMC Sampling...")
            # We use a lower number of draws for notebook execution speed
            self.trace = pm.sample(draws=draws, tune=tune, cores=1, return_inferencedata=True, progressbar=False)
            
        print("Bayesian model fitted successfully.")
        
    def predict_proba(self, X_test):
        if not PYMC_AVAILABLE or self.trace is None:
            # Fallback to random uniform if PyMC fails
            return np.ones((len(X_test), 3)) / 3.0
            
        features_to_use = ['rating_difference', 'home_form_pts_5', 'away_form_pts_5', 'is_neutral']
        X = X_test[features_to_use].copy()
        
        for col in features_to_use:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-9)
            
        X_mat = X.values
        
        # Get posterior means for coefficients
        alpha_mean = self.trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
        beta_mean = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        
        mu = alpha_mean + np.dot(X_mat, beta_mean)
        
        # Softmax
        exp_mu = np.exp(mu - np.max(mu, axis=1, keepdims=True))
        p = exp_mu / np.sum(exp_mu, axis=1, keepdims=True)
        return p
        
    def get_posterior_uncertainty(self, X_sample):
        """Returns mean, lower 95% CI, upper 95% CI, and variance for a single sample."""
        if not PYMC_AVAILABLE or self.trace is None:
            return None
            
        features_to_use = ['rating_difference', 'home_form_pts_5', 'away_form_pts_5', 'is_neutral']
        X = X_sample[features_to_use].copy()
        
        for col in features_to_use:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-9)
            
        X_mat = X.values
        
        alphas = self.trace.posterior['alpha'].values.reshape(-1, 3)
        betas = self.trace.posterior['beta'].values.reshape(-1, X_mat.shape[1], 3)
        
        n_samples = alphas.shape[0]
        n_obs = X_mat.shape[0]
        
        p_samples = np.zeros((n_obs, n_samples, 3))
        
        for i in range(n_samples):
            mu = alphas[i] + np.dot(X_mat, betas[i])
            exp_mu = np.exp(mu - np.max(mu, axis=1, keepdims=True))
            p_samples[:, i, :] = exp_mu / np.sum(exp_mu, axis=1, keepdims=True)
            
        means = p_samples.mean(axis=1)
        variances = p_samples.var(axis=1)
        lower_ci = np.percentile(p_samples, 2.5, axis=1)
        upper_ci = np.percentile(p_samples, 97.5, axis=1)
        
        return means, lower_ci, upper_ci, variances
