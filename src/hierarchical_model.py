import numpy as np

# A simplified mock representation of a Hierarchical model to avoid compiling 
# an extremely complex PyMC model during notebook execution, which would crash or take hours.
# It simulates the outputs of a global -> confederation -> team hierarchy.

class HierarchicalTeamStrengthModel:
    def __init__(self):
        self.team_strengths = {}
        self.confed_strengths = {}
        
    def fit(self, elo_ratings, confederations_map=None):
        print("Fitting Hierarchical Bayesian Team Strength Model...")
        # In a full PyMC model, we would define:
        # global_mu ~ Normal(1500, 100)
        # confed_mu ~ Normal(global_mu, confed_sigma)
        # team_str ~ Normal(confed_mu, team_sigma)
        
        # We simulate the posterior inference here
        global_mu = np.mean(list(elo_ratings.values()))
        
        if confederations_map is None:
            # Fake confederations for demo
            confederations_map = {t: 'UEFA' if i % 2 == 0 else 'CONMEBOL' for i, t in enumerate(elo_ratings.keys())}
            
        confed_groups = {}
        for t, c in confederations_map.items():
            if c not in confed_groups:
                confed_groups[c] = []
            if t in elo_ratings:
                confed_groups[c].append(elo_ratings[t])
                
        for c, ratings in confed_groups.items():
            self.confed_strengths[c] = {
                'mean': np.mean(ratings),
                'sd': np.std(ratings) + 50  # Adding uncertainty
            }
            
        for t, rating in elo_ratings.items():
            confed = confederations_map.get(t, 'UEFA')
            c_mean = self.confed_strengths[confed]['mean']
            
            # Shrinkage effect: pull team rating slightly towards confederation mean
            shrunk_mean = 0.8 * rating + 0.2 * c_mean
            
            self.team_strengths[t] = {
                'mean': shrunk_mean,
                'sd': 30.0 + (1500 / max(rating, 1)) * 10  # Weaker teams have higher uncertainty
            }
            
        print("Hierarchical Model fit complete.")
        
    def get_team_posterior(self, team, n_samples=1000):
        if team not in self.team_strengths:
            return np.random.normal(1500, 100, n_samples)
        
        return np.random.normal(
            self.team_strengths[team]['mean'],
            self.team_strengths[team]['sd'],
            n_samples
        )
