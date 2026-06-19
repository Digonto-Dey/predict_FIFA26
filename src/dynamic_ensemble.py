import numpy as np

class DynamicEnsemble:
    def __init__(self, models):
        self.models = models
        
    def predict_proba(self, X):
        """
        Dynamically weights models based on match context.
        Example: If rating difference is very high, trust XGBoost more.
        If rating difference is low, trust Elo/Ensemble more.
        """
        final_preds = np.zeros((X.shape[0], 3))
        
        for idx in range(X.shape[0]):
            row = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            
            # Simple heuristic: rating_difference is typically at a specific index
            # If using dataframe, we can access by column name
            rating_diff = abs(row['rating_difference']) if 'rating_difference' in X.columns else 0
            
            # Base weights (Equal)
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            
            # Dynamic adjustment
            if rating_diff > 300:
                # High mismatch, ML models might be overconfident, shrink back to a conservative model
                if 'xgb' in weights: weights['xgb'] *= 0.8
                if 'elo' in weights: weights['elo'] *= 1.2
            else:
                # Tight match, ML models capture nuances better
                if 'xgb' in weights: weights['xgb'] *= 1.2
                
            # Normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            
            row_df = X.iloc[[idx]] if hasattr(X, 'iloc') else [X[idx]]
            
            idx_preds = np.zeros(3)
            for name, model in self.models.items():
                preds = model.predict_proba(row_df)[0]
                idx_preds += preds * weights[name]
                
            final_preds[idx] = idx_preds
            
        return final_preds
