import numpy as np

def calculate_confidence_intervals(simulation_results_list):
    """
    Given a list of dictionaries where each dict represents the outcomes 
    (champion, finals, etc.) from one parallel simulation batch,
    calculate the 95% CI.
    """
    # Assuming simulation_results_list is a list of N arrays/dicts representing N simulation runs
    # To get CI, we just find the 2.5th and 97.5th percentiles across the runs
    
    # Restructure into a dictionary of lists
    team_results = {}
    for res in simulation_results_list:
        for team, stages in res.items():
            if team not in team_results:
                team_results[team] = {stage: [] for stage in stages.keys()}
            for stage, val in stages.items():
                team_results[team][stage].append(val)
                
    final_intervals = {}
    for team, stages in team_results.items():
        final_intervals[team] = {}
        for stage, vals in stages.items():
            arr = np.array(vals)
            mean_val = np.mean(arr)
            lower = np.percentile(arr, 2.5)
            upper = np.percentile(arr, 97.5)
            final_intervals[team][stage] = {
                'mean': mean_val,
                'lower_95': lower,
                'upper_95': upper
            }
            
    return final_intervals
