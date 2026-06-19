def apply_stress_scenario(current_ratings, scenario):
    """
    Modifies the baseline ratings according to a stress scenario.
    Returns a new dictionary of modified ratings.
    """
    modified_ratings = current_ratings.copy()
    
    if scenario == "A_top_player_injured":
        # Simulate Argentina losing Messi or France losing Mbappe (drop rating by 50 pts)
        if "Argentina" in modified_ratings:
            modified_ratings["Argentina"] -= 50
        if "France" in modified_ratings:
            modified_ratings["France"] -= 50
            
    elif scenario == "B_team_rating_drops":
        # Global drop in a specific team
        if "Brazil" in modified_ratings:
            modified_ratings["Brazil"] -= 100
            
    elif scenario == "C_unexpected_improvement":
        # African or Asian team overperforms
        if "Morocco" in modified_ratings:
            modified_ratings["Morocco"] += 80
        if "Japan" in modified_ratings:
            modified_ratings["Japan"] += 80
            
    elif scenario == "D_no_home_advantage":
        # Handled in the simulation logic by setting is_neutral=True for everything
        pass
        
    elif scenario == "E_extreme_upsets":
        # Flatten the rating curve (bring everyone closer to 1500)
        for t in modified_ratings:
            diff = modified_ratings[t] - 1500
            modified_ratings[t] = 1500 + (diff * 0.5)
            
    return modified_ratings
