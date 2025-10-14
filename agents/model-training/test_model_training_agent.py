import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from agent_model_selection import ModelSelectionAgent
from agent_hyperparameter_tuning import HyperparameterTuningAgent
from model_evaluation import ModelEvaluationAgent

def main():
    # 📥 Load custom dataset
    print("📥 Loading dataset...")
    df = pd.read_csv('data/raw/agentic_ai_performance_dataset_20250622.csv')

    # 🧹 Drop identifier and encode categorical features
    df = df.drop(columns=['agent_id'])
    df = pd.get_dummies(df, drop_first=True)

    print(df.head())

    # ✅ Define features and target
    X = df.drop(columns='success_rate')
    y = df['success_rate']

    # 🔀 Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 📏 Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    elif best_name == "LinearRegression":
        param_grid = {
            'fit_intercept': [True, False]
        }
    elif best_name == "SVR":
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
