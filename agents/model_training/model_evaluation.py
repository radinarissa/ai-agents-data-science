import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import pandas as pd
import numpy as np

class ModelEvaluationAgent:
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a single regression model and print key metrics."""
        y_pred = model.predict(X_test)
        metrics = {
            "R2 Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
        }

        print("\n📊 Model Evaluation Results:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")

        # Residual Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30, color='purple')
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

        return metrics

    def compare_models(self, models, X_test, y_test):
        """Compare multiple regression models and return a sorted summary DataFrame."""
        print("\n⚖️ Comparing models...\n")
        results = []

        for name, model in models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            results.append({"Model": name, "R2 Score": r2, "RMSE": rmse})


        df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
        print(df.to_string(index=False))

        # Visualization
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x="Model", y="R2 Score", palette="viridis")
        plt.title("Model Comparison (R2 Score)")
        plt.show()

        return df
