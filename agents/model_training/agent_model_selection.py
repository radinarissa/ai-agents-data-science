from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

class ModelSelectionAgent:
    def __init__(self, cv=5, scoring='r2'):
        self.cv = cv
        self.scoring = scoring

    def select_best_model(self, X_train, y_train):
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR()
        }

        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=self.scoring)
            results[name] = scores.mean()
            print(f"✅ {name}: {scores.mean():.4f}")

        best_name = max(results, key=results.get)
        best_model = models[best_name]
        print(f"🏆 Best Model: {best_name} (Score: {results[best_name]:.4f})")
        return best_name, best_model, results
