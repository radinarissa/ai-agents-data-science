# model_evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
import numpy as np

class ModelEvaluationAgent:
    def __init__(self, average='macro'):
        self.average = average  # Used for multi-class F1, precision, recall

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a single model and print key metrics."""
        y_pred = model.predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average=self.average, zero_division=0),
            "Recall": recall_score(y_test, y_pred, average=self.average, zero_division=0),
            "F1": f1_score(y_test, y_pred, average=self.average, zero_division=0)
        }

        print("\n📊 Model Evaluation Results:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")

        print("\n📄 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return metrics

    def compare_models(self, models, X_test, y_test):
        """Compare multiple trained models and return a sorted summary DataFrame."""
        print("\n⚖️ Comparing models...\n")
        results = []

        for name, model in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average=self.average, zero_division=0)
            results.append({"Model": name, "Accuracy": acc, "F1": f1})

        df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
        print(df.to_string(index=False))

        # Visualization
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x="Model", y="F1")
        plt.title("Model Comparison (F1 Score)")
        plt.show()

        return df
