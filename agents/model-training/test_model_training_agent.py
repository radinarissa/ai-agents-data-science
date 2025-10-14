# test_model_training_agent.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from agent_model_selection import ModelSelectionAgent
from agent_hyperparameter_tuning import HyperparameterTuningAgent
from model_evaluation import ModelEvaluationAgent

def main():
    # Load dataset
    print("📥 Loading dataset...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1️⃣ Model selection
    print("\n=== MODEL SELECTION ===")
    selector = ModelSelectionAgent()
    best_name, best_model, results = selector.select_best_model(X_train, y_train)

    # 2️⃣ Hyperparameter tuning
    print("\n=== HYPERPARAMETER TUNING ===")
    tuner = HyperparameterTuningAgent(model=best_model)

    if best_name == "RandomForest":
        param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, None]
    }
    elif best_name == "LogisticRegression":
     param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }
    elif best_name == "SVC":
        param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf']
    }
    else:
        raise ValueError(f"Unknown model: {best_name}")

    best_tuned_model = tuner.grid_search(X_train, y_train, param_grid)


    # 3️⃣ Model evaluation
    print("\n=== MODEL EVALUATION ===")
    evaluator = ModelEvaluationAgent()
    metrics = evaluator.evaluate_model(best_tuned_model, X_test, y_test)

    print("\n✅ Final Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
