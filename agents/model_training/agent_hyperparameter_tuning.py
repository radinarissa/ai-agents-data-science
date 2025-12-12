import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class HyperparameterTuningAgent:
    def __init__(self, model=None, cv=3, scoring='r2'):
        self.model = model or RandomForestRegressor()
        self.cv = cv
        self.scoring = scoring

    def _evaluate_best(self, search, X_train, y_train):
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_train)
        metrics = {
            "Best Params": search.best_params_,
            "R2 Score": r2_score(y_train, y_pred),
            "RMSE": mean_squared_error(y_train, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_train, y_pred)
        }
        return best_model, metrics

    def grid_search(self, X_train, y_train, param_grid):
        print("🔧 Running Grid Search...")
        start = time.time()
        grid = GridSearchCV(self.model, param_grid, cv=self.cv, scoring=self.scoring)
        grid.fit(X_train, y_train)
        elapsed = time.time() - start
        best_model, metrics = self._evaluate_best(grid, X_train, y_train)
        metrics["Time (s)"] = elapsed
        print(f"✅ Best Params: {grid.best_params_}")
        return best_model, metrics

    def random_search(self, X_train, y_train, param_distributions, n_iter=20):
        print("🎲 Running Random Search...")
        start = time.time()
        random_search = RandomizedSearchCV(
            self.model, param_distributions, n_iter=n_iter,
            cv=self.cv, scoring=self.scoring, random_state=42
        )
        random_search.fit(X_train, y_train)
        elapsed = time.time() - start
        best_model, metrics = self._evaluate_best(random_search, X_train, y_train)
        metrics["Time (s)"] = elapsed
        print(f"✅ Best Params: {random_search.best_params_}")
        return best_model, metrics

    def bayesian_optimization(self, X_train, y_train, search_spaces, n_iter=25):
        print("🧠 Running Bayesian Optimization...")
        start = time.time()
        bayes = BayesSearchCV(
            self.model, search_spaces, n_iter=n_iter,
            cv=self.cv, scoring=self.scoring, random_state=42
        )
        bayes.fit(X_train, y_train)
        elapsed = time.time() - start
        best_model, metrics = self._evaluate_best(bayes, X_train, y_train)
        metrics["Time (s)"] = elapsed
        print(f"✅ Best Params: {bayes.best_params_}")
        return best_model, metrics