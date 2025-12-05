import pandas as pd
import sys, os, json
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
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

    # ✅ Guard check for target column
    if "success_rate" not in df.columns:
        raise ValueError("Target column 'success_rate' missing in processed dataset")
    
    X = df.drop(columns=["success_rate"])
    y = df["success_rate"]

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (needed for linear models and SVR)
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
        # RandomForest doesn’t need scaled input
        tuned_model, tuning_metrics = tuner.grid_search(X_train, y_train, param_grid)
    elif best_name == "SVR":
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        tuned_model, tuning_metrics = tuner.grid_search(X_train_scaled, y_train, param_grid)
    else:
        param_grid = {}
        tuned_model = best_model
        tuning_metrics = {"note": "No tuning performed"}

    if param_grid:
        print(f"Tuning metrics: {tuning_metrics}")

    # -------------------------------
    # Experiment 6: Residual Analysis
    # -------------------------------

    print("\n=== Експеримент 6: Residual Analysis – Качество на предсказанията ===")
    evaluator = ModelEvaluationAgent()

    # Use scaled input for models that need it
    if best_name == "RandomForest":
        final_metrics = evaluator.evaluate_model(tuned_model, X_test, y_test)
    else:
        final_metrics = evaluator.evaluate_model(tuned_model, X_test_scaled, y_test)

    residuals = y_test - tuned_model.predict(X_test if best_name=="RandomForest" else X_test_scaled)
    predicted = tuned_model.predict(X_test if best_name=="RandomForest" else X_test_scaled)

    # Residual Distribution
    plt.figure(figsize=(8,6))
    sns.histplot(residuals, kde=True, color="purple", edgecolor="black")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig("experiments/results/experiment6_residual_distribution.png", dpi=300)
    plt.close()

    # Q-Q Plot
    sm.qqplot(residuals, line='45', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.savefig("experiments/results/experiment6_qqplot.png", dpi=300)
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=predicted, y=residuals, color="purple", edgecolor="black", alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.savefig("experiments/results/experiment6_residuals_vs_predicted.png", dpi=300)
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=predicted, color="blue", edgecolor="black", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.savefig("experiments/results/experiment6_actual_vs_predicted.png", dpi=300)
    plt.close()

    print("\n✅ Experiments complete. Results saved in experiments/results/")

    # ✅ Save all results for 4-6
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/all_results_4_5_6.json", "w") as f:
        json.dump({
            "selection_results": results,
            "tuning_metrics": tuning_metrics,
            "final_metrics": final_metrics
        }, f, indent=2)

    # -------------------------------
    # Experiment 10: Visualization of Processed Dataset
    # -------------------------------

    print("\n=== Експеримент 10: Visualization of Processed Dataset ===")

    print("First 5 rows of the processed dataset:")
    print(df.head())

    print("\nColumn information:")
    df.info()

    print("\nDescriptive statistics:")
    print(df.describe())

    # Ensure results directory exists
    os.makedirs("experiments/results", exist_ok=True)

    # Histogram of numeric features
    df.hist(figsize=(10, 6), bins=10)
    plt.suptitle("Distribution of Numeric Features", fontsize=14)
    plt.tight_layout()
    plt.savefig("experiments/results/histograms.png", dpi=300)
    plt.close()

    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(8, 6))
    plt.imshow(numeric_df.corr(), cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=45)
    plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
    plt.title("Correlation Matrix (Numeric Features Only)")
    plt.tight_layout()
    plt.savefig("experiments/results/correlation_matrix.png", dpi=300)
    plt.close()

    print("✅ Visualization complete. Figures saved in experiments/results/")

if __name__ == "__main__":
    run_experiments()
