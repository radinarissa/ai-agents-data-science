import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import pandas as pd
import numpy as np
from scipy.stats import shapiro

class ModelEvaluationAgent:
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a single regression model and print key metrics + residual analysis."""
        y_pred = model.predict(X_test)

        # Основни метрики
        metrics = {
            "R2 Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
        }

        # Остатъци
        residuals = y_test - y_pred
        residual_stats = {
            "Mean Residual": np.mean(residuals),
            "Std Residual": np.std(residuals),
            "Skewness": pd.Series(residuals).skew(),
            "Kurtosis": pd.Series(residuals).kurt()
        }
        metrics.update(residual_stats)

        # Shapiro-Wilk тест
        stat, p_value = shapiro(residuals)
        metrics["Shapiro-Wilk Statistic"] = stat
        metrics["Shapiro-Wilk p-value"] = p_value

        # Печат
        print("\n📊 Model Evaluation Results:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")

        # Визуализации - SAVE instead of SHOW (non-blocking)
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30, color='purple')
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.savefig("results/residual_distribution.png", dpi=100, bbox_inches='tight')
        plt.close()  # Close instead of show

        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.savefig("results/actual_vs_predicted.png", dpi=100, bbox_inches='tight')
        plt.close()  # Close instead of show

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

        # Визуализация - SAVE instead of SHOW
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x="Model", y="R2 Score", palette="viridis")
        plt.title("Model Comparison (R2 Score)")
        plt.savefig("results/model_comparison.png", dpi=100, bbox_inches='tight')
        plt.close()

        return df