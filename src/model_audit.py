import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss

def perform_data_leakage_audit(df, train_mask, val_mask, test_mask):
    """Checks if there's any overlap between train, val, and test dates/indices."""
    train_dates = df.loc[train_mask, 'date']
    val_dates = df.loc[val_mask, 'date']
    test_dates = df.loc[test_mask, 'date']
    
    leakage_val = train_dates.isin(val_dates).sum() > 0 or val_dates.isin(train_dates).sum() > 0
    leakage_test = train_dates.isin(test_dates).sum() > 0 or test_dates.isin(train_dates).sum() > 0
    
    print(f"Data Leakage Audit - Train vs Val overlap: {leakage_val}")
    print(f"Data Leakage Audit - Train vs Test overlap: {leakage_test}")
    
    if df[train_mask]['date'].max() >= df[val_mask]['date'].min():
        print("WARNING: Train dates overlap or are newer than Val dates. Potential Future Contamination.")
    else:
        print("Pass: Train data precedes Validation data properly.")
        
def evaluate_calibration(y_true, y_probs, title="Calibration Plot"):
    """Evaluates probability calibration."""
    # Since y_true is multiclass (0, 1, 2), we check one-vs-rest for simplicity (e.g., Home Win)
    from sklearn.calibration import calibration_curve
    y_true_binary = (y_true == 2).astype(int)
    y_probs_home = y_probs[:, 2]
    
    prob_true, prob_pred = calibration_curve(y_true_binary, y_probs_home, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label="Home Win")
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.title(title)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    brier = brier_score_loss(y_true_binary, y_probs_home)
    print(f"Brier Score (Home Win): {brier:.4f}")
    return brier

def run_full_audit(df, train_mask, val_mask, test_mask, models, X_val, y_val):
    print("--- STARTING MODEL AUDIT ---")
    perform_data_leakage_audit(df, train_mask, val_mask, test_mask)
    
    for name, model in models.items():
        print(f"\\nEvaluating Calibration for {name}:")
        probs = model.predict_proba(X_val)
        evaluate_calibration(y_val, probs, title=f"Calibration: {name}")
    print("--- AUDIT COMPLETE ---")
