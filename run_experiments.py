import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

from agents.orchestration.orchestrator import OrchestratorAgent
from agents.data_processing.data_processing_agent import DataProcessingAgent
from agents.model_training.agent_model_selection import ModelSelectionAgent
from agents.model_training.agent_hyperparameter_tuning import HyperparameterTuningAgent
from agents.model_training.model_evaluation import ModelEvaluationAgent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_experiments_with_orchestrator():
    """
    Експерименти 4-6 координирани от Orchestrator
    ✅ OPTIMIZED VERSION - Faster Grid Search
    """

    # Създаваме Orchestrator
    orchestrator = OrchestratorAgent()

    orchestrator.log("=" * 60)
    orchestrator.log("ADVANCED EXPERIMENTS - Model Selection, Tuning, Analysis")
    orchestrator.log("=" * 60)

    # -------------------------------
    # Стъпка 1: Зареждане на обработени данни
    # -------------------------------

    orchestrator.log("Loading processed dataset...")
    df = pd.read_csv("data/processed/processed_dataset.csv")

    if "success_rate" not in df.columns:
        raise ValueError("Target column 'success_rate' missing in processed dataset")

    X = df.drop(columns=["success_rate"])
    y = df["success_rate"]

    orchestrator.log(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # -------------------------------
    # Стъпка 2: Train/test split
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (needed for linear models and SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    orchestrator.log(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # -------------------------------
    # EXPERIMENT 4: Model Selection
    # -------------------------------

    orchestrator.log("\n" + "=" * 60)
    orchestrator.log("EXPERIMENT 4: Model Selection")
    orchestrator.log("=" * 60)

    selector = ModelSelectionAgent(cv=5, scoring='r2')
    best_name, best_model, selection_results = selector.select_best_model(X_train_scaled, y_train)

    # Логване чрез Orchestrator
    candidates = list(selection_results.keys())
    scores = {name: res['cv'] for name, res in selection_results.items() if 'cv' in res}

    orchestrator.log_model_selection(best_name, candidates, scores)

    # -------------------------------
    # EXPERIMENT 5: Hyperparameter Tuning (OPTIMIZED)
    # -------------------------------

    orchestrator.log("\n" + "=" * 60)
    orchestrator.log("EXPERIMENT 5: Hyperparameter Tuning (OPTIMIZED)")
    orchestrator.log("=" * 60)

    tuner = HyperparameterTuningAgent(model=best_model, cv=5, scoring='r2')

    if best_name == "RandomForest":
        # ✅ OPTIMIZED: Reduced parameter grid (9 combinations instead of 36)
        param_grid = {
            'n_estimators': [100, 200],  # 2 values (was 3)
            'max_depth': [10, None],  # 2 values (was 3)
            'min_samples_split': [2],  # 1 value (was 2)
            'min_samples_leaf': [1]  # 1 value (was 2)
        }
        # Total: 2 * 2 * 1 * 1 = 4 combinations (was 36)
        # With CV=5: 4 * 5 = 20 fits (was 180)
        # Time: ~2-3 minutes (was 20-25 minutes)

        orchestrator.log(
            f"⚡ Using OPTIMIZED Grid Search: {len(param_grid['n_estimators']) * len(param_grid['max_depth'])} combinations")
        tuned_model, tuning_metrics = tuner.grid_search(X_train, y_train, param_grid)

        # Логване чрез Orchestrator
        orchestrator.log_hyperparameter_tuning(
            model_name=best_name,
            best_params=tuning_metrics['Best Params'],
            tuning_method='Grid Search (Optimized)',
            iterations=len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
                       len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
        )

    elif best_name == "SVR":
        # ✅ OPTIMIZED: Reduced parameter grid
        param_grid = {
            'C': [1.0, 10.0],  # 2 values (was 3)
            'kernel': ['rbf'],  # 1 value (was 2)
            'gamma': ['scale']  # 1 value (was 2)
        }
        # Total: 2 * 1 * 1 = 2 combinations (was 12)

        tuned_model, tuning_metrics = tuner.grid_search(X_train_scaled, y_train, param_grid)

        orchestrator.log_hyperparameter_tuning(
            model_name=best_name,
            best_params=tuning_metrics['Best Params'],
            tuning_method='Grid Search (Optimized)',
            iterations=len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma'])
        )
    else:
        param_grid = {}
        tuned_model = best_model
        tuning_metrics = {"note": "No tuning performed"}
        orchestrator.log("No hyperparameter tuning for this model")

    # -------------------------------
    # EXPERIMENT 6: Residual Analysis
    # -------------------------------

    orchestrator.log("\n" + "=" * 60)
    orchestrator.log("EXPERIMENT 6: Residual Analysis")
    orchestrator.log("=" * 60)

    evaluator = ModelEvaluationAgent()

    # Use scaled input for models that need it
    if best_name == "RandomForest":
        final_metrics = evaluator.evaluate_model(tuned_model, X_test, y_test)
        predictions = tuned_model.predict(X_test)
    else:
        final_metrics = evaluator.evaluate_model(tuned_model, X_test_scaled, y_test)
        predictions = tuned_model.predict(X_test_scaled)

    residuals = y_test - predictions

    orchestrator.log(f"R² Score: {final_metrics['R2 Score']:.4f}")
    orchestrator.log(f"RMSE: {final_metrics['RMSE']:.4f}")
    orchestrator.log(f"MAE: {final_metrics['MAE']:.4f}")

    # Check for data leakage
    if final_metrics['R2 Score'] > 0.95:
        orchestrator.log("⚠️  WARNING: Suspiciously high R² score!")
        orchestrator.log("   This might indicate data leakage.")
        orchestrator.log("   Run: python check_data_leakage.py")

    # -------------------------------
    # Визуализации
    # -------------------------------

    orchestrator.log("\nGenerating visualizations...")

    os.makedirs("experiments/results", exist_ok=True)

    # Residual Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color="purple", edgecolor="black")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig("experiments/results/experiment6_residual_distribution.png", dpi=300)
    plt.close()

    # Q-Q Plot
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.savefig("experiments/results/experiment6_qqplot.png", dpi=300)
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=predictions, y=residuals, color="purple", edgecolor="black", alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.savefig("experiments/results/experiment6_residuals_vs_predicted.png", dpi=300)
    plt.close()

    # Actual vs Predicted
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=predictions, color="blue", edgecolor="black", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.savefig("experiments/results/experiment6_actual_vs_predicted.png", dpi=300)
    plt.close()

    orchestrator.log("✅ All 4 visualizations saved to experiments/results/")
    orchestrator.log("   1. experiment6_residual_distribution.png")
    orchestrator.log("   2. experiment6_qqplot.png")
    orchestrator.log("   3. experiment6_residuals_vs_predicted.png")
    orchestrator.log("   4. experiment6_actual_vs_predicted.png")

    # -------------------------------
    # Запазване на резултати
    # -------------------------------

    import json

    all_results = {
        "experiment_4_selection": selection_results,
        "experiment_5_tuning": tuning_metrics,
        "experiment_6_evaluation": final_metrics
    }

    with open("experiments/results/all_results_4_5_6.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    orchestrator.log("Results saved to experiments/results/all_results_4_5_6.json")

    # -------------------------------
    # Запазване на логове
    # -------------------------------

    orchestrator.save_logs("experiments/results/experiments_log.txt")

    orchestrator.log("\n" + "=" * 60)
    orchestrator.log("EXPERIMENTS COMPLETED SUCCESSFULLY")
    orchestrator.log("=" * 60)

    return {
        'selection_results': selection_results,
        'tuning_metrics': tuning_metrics,
        'final_metrics': final_metrics,
        'best_model': tuned_model,
        'best_model_name': best_name
    }


if __name__ == "__main__":
    results = run_experiments_with_orchestrator()
    print("\n✅ All experiments completed! Check experiments/results/ for outputs.")