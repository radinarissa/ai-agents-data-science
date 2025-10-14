# agent_model_selection.py

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

class ModelSelectionAgent:
    def __init__(self, models=None, cv=5, scoring='accuracy'):
        # Define default models if none provided
        self.models = models or {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC()
        }
        self.cv = cv
        self.scoring = scoring

    def select_best_model(self, X_train, y_train):
        results = {}
        print("🔍 Starting model selection...")
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=self.scoring)
            results[name] = np.mean(scores)
            print(f"✅ {name}: {results[name]:.4f}")

        # Find best model
        best_model_name = max(results, key=results.get)
        best_model = self.models[best_model_name]

        print(f"\n🏆 Best Model: {best_model_name} (Score: {results[best_model_name]:.4f})")
        best_model.fit(X_train, y_train)
        return best_model_name, best_model, results
