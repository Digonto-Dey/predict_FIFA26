# FIFA World Cup 2026 Prediction Project

## Overview
This repository contains a comprehensive, research-grade machine learning and Bayesian forecasting system designed to predict the outcomes of the FIFA World Cup 2026 tournament. The project evaluates historical international football data spanning back to 1872 and builds a progressive pipeline involving an advanced rating system, machine learning match prediction, and a Monte Carlo tournament simulation engine.

The end-to-end process is divided into 7 distinct phases, ultimately producing calibrated match probabilities, champion probabilities, and dynamic confidence intervals for every team in the tournament.

## Project Architecture & Methodology

### Phase 1: Data Cleaning and Exploration
- **Data Ingestion**: Processes historical matches (`results.csv`) and goalscorer data (`goalscorers.csv`).
- **Standardization**: Converts dates and replaces obsolete historical country names (e.g., Zaire, Yugoslavia) using `former_names.csv`.
- **Validation**: Checks for data quality issues such as negative scores, missing values, and invalid tournaments, generating summary visualizations.

### Phase 2: Match Result Target Variable
- **Target Generation**: Defines the `result` classification target (2 = Home Win, 1 = Draw, 0 = Away Win).
- **Match Features**: Creates base statistics including `goal_difference`, `total_goals`, and historical outcome distributions.

### Phase 3: Advanced Football Rating System (Enhanced Elo Framework)
- **Beyond Classical Elo**: Implements a highly customized, hybrid Elo rating system.
- **Home Advantage**: Empirically estimated from historical data (neutral venues are explicitly handled).
- **Tournament Importance Weighting**: Differentiates weights for World Cups, Continental tournaments, Qualifiers, and Friendlies.
- **Goal Difference Multipliers**: Modifies rating updates based on the margin of victory.
- **Dynamic K-Factor & Time Decay**: Adapts the learning rate based on match importance and applies exponential time decay to favor recent form.
- **Explicit Draw Modeling**: Enhances the standard Elo expected score to directly output P(Home Win), P(Draw), and P(Away Win).

### Phase 4: Machine Learning Prediction Engine
- **Feature Engineering**: Constructs pre-match features including Elo probabilities, form (last 5/10 matches), rolling goal averages, and opponent-adjusted strength metrics.
- **Models**: Evaluates Multinomial Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and Neural Networks (MLP).
- **Validation**: Employs chronological time-series rolling-origin validation to strictly prevent data leakage.
- **Calibration**: Uses Isotonic Calibration, Platt Scaling, and Temperature Scaling to ensure output probabilities are statistically reliable.
- **Interpretability**: Generates SHAP values and permutation importances to explain model decisions.

### Phase 5: Tournament Simulation Engine
- **Monte Carlo Simulator**: Runs 100,000 parallel tournament simulations for the FIFA 2026 format.
- **Group Stage**: Samples match outcomes from ML-calibrated probabilities, implements official FIFA tiebreakers, and handles top-2 plus best third-place qualification rules.
- **Knockout Stage**: Handles extra time and penalty shootout approximations for matches that cannot end in a draw.
- **Outputs**: Computes the exact probability of every team reaching each stage of the tournament (Group Qualification -> Round of 32 -> ... -> Champion).

### Phase 6: Ensemble Intelligence Engine
- **Model Stacking & Averaging**: Combines predictions from multiple Level-1 models (Elo, XGBoost, LightGBM, etc.) using Level-2 meta-models.
- **Dynamic Weighting**: Adjusts ensemble weights based on match context (e.g., tournament importance, rating gap).
- **Benchmarking**: Compares Equal Weight, Validation Weight, Bayesian Model Averaging, and Advanced Stacking approaches.

### Phase 7: Bayesian Football Forecasting
- **Uncertainty Quantification**: Implements Bayesian Multinomial Logistic Regression (via PyMC) to output posterior distributions rather than single point estimates.
- **Hierarchical Modeling**: Models team strength hierarchically (Global -> Confederation -> National Team).
- **Stress Testing**: Analyzes forecast stability against hypothetical scenarios (e.g., top player injuries, major rating shocks).
- **Final Output**: Provides 95% confidence intervals for all tournament progression probabilities.

## Core Technologies
- **Python 3.11+**
- **Data Manipulation**: `pandas`, `numpy`, `scipy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `tensorflow`, `keras`
- **Optimization**: `optuna` (Hyperparameter tuning)
- **Interpretability**: `shap`
- **Visualizations**: `matplotlib`, `seaborn`, `plotly`

## Repository Structure
```
predict_fifa2026/
├── data/                       # Raw datasets and processed/cleaned target CSVs
├── src/                        # Reusable Python modules (ensemble, bayesian_models, simulation_extensions, etc.)
├── 01_data_cleaning.py         # Phase 1 scripts
├── 01_data_cleaning.ipynb      # Phase 1 notebook
├── 02_match_targets.py         # Phase 2 scripts
├── 02_match_targets.ipynb      # Phase 2 notebook
├── fifa2026_prediction.ipynb   # MASTER Research Notebook (Contains end-to-end pipeline starting from Phase 3)
├── requirements.txt            # Python dependencies
├── former_names.csv            # Historical country names mapping
├── goalscorers.csv             # Goalscorers dataset
├── results.csv                 # International match results dataset
└── README.md                   # Project documentation
```

## Setup Instructions
1. **Clone the repository**:
   Ensure you have the project folder setup on your local machine.
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   - Execute the data cleaning and target creation scripts (`01_data_cleaning.py` and `02_match_targets.py`) to prepare the foundational datasets.
   - Open `fifa2026_prediction.ipynb`. This is the single source of truth for the project. Run this notebook from top to bottom to train the models, run the Bayesian analysis, and execute the 100,000-iteration Monte Carlo World Cup simulation.

## Evaluation & Success Criteria
The primary metric for this project is **Log Loss** and **Calibration Error**, ensuring that our predicted probabilities reflect reality. The final model is rigorously tested to out-perform standard Elo systems and generalizing perfectly to unseen, future international match windows.
