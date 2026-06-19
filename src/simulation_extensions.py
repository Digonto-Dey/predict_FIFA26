from collections import defaultdict
import random
from itertools import combinations

def run_tournament_monte_carlo(teams, groups, predict_match_fn, n_sims=1000, seed=42):
    """
    Runs a Monte Carlo simulation of the tournament using a custom prediction function.
    Returns the results dictionary.
    """
    random.seed(seed)
    results_tracker = defaultdict(lambda: {
        'Group': 0, 'R32': 0, 'R16': 0, 'QF': 0, 'SF': 0, 'Final': 0, 'Champion': 0
    })
    
    for sim in range(n_sims):
        group_standings = {}
        for g, g_teams in groups.items():
            pts = {t: 0 for t in g_teams}
            for t1, t2 in combinations(g_teams, 2):
                res = predict_match_fn(t1, t2, allow_draw=True)
                if res == 2: pts[t1] += 3
                elif res == 1: pts[t1] += 1; pts[t2] += 1
                else: pts[t2] += 3
            ranked = sorted(pts.items(), key=lambda x: x[1], reverse=True)
            group_standings[g] = [r[0] for r in ranked]
            
        r32_teams = []
        third_places = []
        for g, ranked in group_standings.items():
            r32_teams.extend(ranked[:2])
            third_places.append(ranked[2])
            
        random.shuffle(third_places)
        r32_teams.extend(third_places[:8])
        for t in r32_teams: results_tracker[t]['R32'] += 1
        
        def run_knockout_round(t_list, round_name):
            next_round = []
            for i in range(0, len(t_list), 2):
                res = predict_match_fn(t_list[i], t_list[i+1], allow_draw=False)
                winner = t_list[i] if res == 2 else t_list[i+1]
                next_round.append(winner)
            for t in next_round: results_tracker[t][round_name] += 1
            return next_round
            
        r16_teams = run_knockout_round(r32_teams, 'R16')
        qf_teams = run_knockout_round(r16_teams, 'QF')
        sf_teams = run_knockout_round(qf_teams, 'SF')
        final_teams = run_knockout_round(sf_teams, 'Final')
        champion = run_knockout_round(final_teams, 'Champion')
        
    return results_tracker
