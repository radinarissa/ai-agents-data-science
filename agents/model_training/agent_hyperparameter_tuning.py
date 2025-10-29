from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV 
import numpy as np

class HyperparameterTuningAgent:
    def __init__(self, model=None, cv=3, scoring='r2'):
        self.model = model or RandomForestRegressor()
        self.cv = cv
        self.scoring = scoring

    def grid_search(self, X_train, y_train, param_grid):
        print("🔧 Running Grid Search...")
        grid = GridSearchCV(self.model, param_grid, cv=self.cv, scoring=self.scoring)
        grid.fit(X_train, y_train)
        print(f"✅ Best Params: {grid.best_params_}")
        return grid.best_estimator_

    def random_search(self, X_train, y_train, param_distributions, n_iter=20):
        print("🎲 Running Random Search...")
        random_search = RandomizedSearchCV(
            self.model, param_distributions, n_iter=n_iter, cv=self.cv, scoring=self.scoring, random_state=42
        )
        random_search.fit(X_train, y_train)
        print(f"✅ Best Params: {random_search.best_params_}")
        return random_search.best_estimator_

    def bayesian_optimization(self, X_train, y_train, search_spaces, n_iter=25):
        print("🧠 Running Bayesian Optimization...")
        bayes = BayesSearchCV(
            self.model, search_spaces, n_iter=n_iter, cv=self.cv, scoring=self.scoring, random_state=42
        )
        bayes.fit(X_train, y_train)
        print(f"✅ Best Params: {bayes.best_params_}")
        return bayes.best_estimator_