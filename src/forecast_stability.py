import numpy as np

def measure_forecast_variance(baseline_probs, perturbed_probs):
    """
    Measures how much forecasts change when training data changes.
    baseline_probs: Dict of {team: win_probability}
    perturbed_probs: Dict of {team: win_probability}
    """
    variances = {}
    for team in baseline_probs:
        p_base = baseline_probs.get(team, 0)
        p_pert = perturbed_probs.get(team, 0)
        
        # Simple variance measure
        variances[team] = (p_base - p_pert) ** 2
        
    overall_stability_score = 1.0 - np.mean(list(variances.values()))
    return overall_stability_score, variances
