import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.model_training.agent_model_selection import ModelSelectionAgent
from agents.model_training.agent_hyperparameter_tuning import HyperparameterTuningAgent
from agents.model_training.model_evaluation import ModelEvaluationAgent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_experiments():
    print("⚗️ RUNNING EXPERIMENTS – Model Selection, Tuning, Residual Analysis")

    # Step 1: Load processed dataset
    df = pd.read_csv("data/processed/processed_dataset.csv")
    X = df.drop(columns=["success_rate"])
    y = df["success_rate"]

    # Step 2: Train/test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------
    # Experiment 4: Model Selection
    # -------------------------------
    print("\n=== Експеримент 4: Model Selection – Сравнителен анализ ===")
    selector = ModelSelectionAgent(cv=5, scoring='r2')
    best_name, best_model, results = selector.select_best_model(X_train_scaled, y_train)

    # -------------------------------
    # Experiment 5: Hyperparameter Tuning
    # -------------------------------
    print("\n=== Експеримент 5: Hyperparameter Tuning – A/B Тестове ===")
    tuner = HyperparameterTuningAgent(model=best_model, cv=5, scoring='r2')

    if best_name == "RandomForest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif best_name == "SVR":
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    else:
        param_grid = {}

    if param_grid:
        tuned_model, tuning_metrics = tuner.grid_search(X_train_scaled, y_train, param_grid)
    else:
        tuned_model = best_model
        tuning_metrics = {"note": "No tuning performed"}

    # -------------------------------
    # Experiment 6: Residual Analysis
    # -------------------------------
    print("\n=== Експеримент 6: Residual Analysis – Качество на предсказанията ===")
    evaluator = ModelEvaluationAgent()
    final_metrics = evaluator.evaluate_model(tuned_model, X_test_scaled, y_test)

    print("\n✅ Experiments complete. Results saved in experiments/results/")

if __name__ == "__main__":
    run_experiments()
