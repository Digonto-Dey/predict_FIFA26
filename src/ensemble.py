import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

class SimpleEnsemble:
    def __init__(self, models):
        self.models = models
        self.weights = None
        
    def fit_equal_weights(self):
        self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
    def fit_validation_weights(self, X_val, y_val):
        losses = {}
        for name, model in self.models.items():
            preds = model.predict_proba(X_val)
            losses[name] = log_loss(y_val, preds)
        
        # Inverse loss weighting
        inv_losses = {name: 1.0 / loss for name, loss in losses.items()}
        total_inv = sum(inv_losses.values())
        self.weights = {name: inv / total_inv for name, inv in inv_losses.items()}
        
    def predict_proba(self, X):
        if self.weights is None:
            self.fit_equal_weights()
            
        final_preds = None
        for name, model in self.models.items():
            preds = model.predict_proba(X)
            if final_preds is None:
                final_preds = preds * self.weights[name]
            else:
                final_preds += preds * self.weights[name]
        return final_preds
