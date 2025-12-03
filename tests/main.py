import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

from agents.model_training.agent_model_selection import ModelSelectionAgent
from agents.model_training.agent_hyperparameter_tuning import HyperparameterTuningAgent
from agents.model_training.model_evaluation import ModelEvaluationAgent

Path("experiments/results").mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    print("📥 LOADING AND PREPARING DATASET")
    
    df = pd.read_csv('data/raw/agentic_ai_performance_dataset_20250622.csv')
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    df = df.drop(columns=['agent_id'])
    df = pd.get_dummies(df, drop_first=True)
    
    print(f"✅ After preprocessing: {df.shape[1]} features")
    print(f"\nFirst few rows:")
    print(df.head())
    X = df.drop(columns='success_rate')
    y = df['success_rate']
    
    print(f"\n✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"✅ Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    return X, y

def experiment_baseline_model_selection(X, y):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: MODEL SELECTION - COMPARATIVE ANALYSIS")
    print("=" * 70 + "\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("🔍 Testing multiple models with cross-validation...\n")
    selector = ModelSelectionAgent(cv=5, scoring='r2')
    best_name, best_model, results = selector.select_best_model(X_train_scaled, y_train)
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    
    test_metrics = {
        "Model": best_name,
        "CV R² (mean)": results[best_name],
        "Test RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test R²": r2_score(y_test, y_pred)
    }
    
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Test RMSE: {test_metrics['Test RMSE']:.4f}")
    print(f"   Test R²: {test_metrics['Test R²']:.4f}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    models = list(results.keys())
    scores = list(results.values())
    colors = ['#1f77b4' if m != best_name else '#ff7f0e' for m in models]
    
    ax.barh(models, scores, color=colors, alpha=0.8)
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_title('Model Comparison (Cross-Validation)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/results/experiment4_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    pd.DataFrame([test_metrics]).to_csv("experiments/results/experiment4_results.csv", index=False)
    
    return best_name, best_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def experiment_hyperparameter_tuning(best_name, best_model, X_train, y_train, X_test, y_test):
    print("EXPERIMENT 5: HYPERPARAMETER TUNING")
    
    tuner = HyperparameterTuningAgent(model=best_model, cv=5, scoring='r2')
    if best_name == "RandomForest":
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_name == "LinearRegression":
        param_grid = {
            'fit_intercept': [True, False]
        }
    elif best_name == "SVR":
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    else:
        print(f"⚠️  No tuning parameters defined for {best_name}, skipping...")
        return best_model
    
    print(f"🔧 Tuning {best_name} with Grid Search...")
    print(f"   Search space: {param_grid}\n")
    
    start_time = time.time()
    best_tuned_model = tuner.grid_search(X_train, y_train, param_grid)
    elapsed = time.time() - start_time
    
    y_pred_tuned = best_tuned_model.predict(X_test)
    tuned_rmse = mean_squared_error(y_test, y_pred_tuned) ** 0.5
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    
    y_pred_original = best_model.fit(X_train, y_train).predict(X_test)
    original_rmse = mean_squared_error(y_test, y_pred_original) ** 0.5
    original_r2 = r2_score(y_test, y_pred_original)
    
    print(f"\n📊 Tuning Results:")
    print(f"   Original RMSE: {original_rmse:.4f} | Tuned RMSE: {tuned_rmse:.4f}")
    print(f"   Original R²: {original_r2:.4f} | Tuned R²: {tuned_r2:.4f}")
    print(f"   Improvement: {((original_rmse - tuned_rmse) / original_rmse * 100):.2f}% reduction in RMSE")
    print(f"   Time elapsed: {elapsed:.2f} seconds")
    
    tuning_results = {
        "Model": best_name,
        "Original RMSE": original_rmse,
        "Tuned RMSE": tuned_rmse,
        "Original R²": original_r2,
        "Tuned R²": tuned_r2,
        "Improvement (%)": (original_rmse - tuned_rmse) / original_rmse * 100,
        "Tuning Time (s)": elapsed
    }
    pd.DataFrame([tuning_results]).to_csv("experiments/results/experiment5_tuning_results.csv", index=False)
    
    return best_tuned_model

def experiment_residual_analysis(model, X_test, y_test, model_name):
    print("EXPERIMENT 6: RESIDUAL ANALYSIS - MODEL QUALITY")

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("📊 Final Model Performance:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    
    print(f"\n📈 Residual Statistics:")
    print(f"   Mean: {np.mean(residuals):.6e}")
    print(f"   Std: {np.std(residuals):.4f}")
    print(f"   Skewness: {stats.skew(residuals):.4f}")
    print(f"   Kurtosis: {stats.kurtosis(residuals):.4f}")
    
    if len(residuals) > 3:
        stat, p_value = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"\n🔬 Shapiro-Wilk Test:")
        print(f"   Statistic: {stat:.4f}")
        print(f"   p-value: {p_value:.4f}")
        print(f"   {'✅ Residuals appear normally distributed' if p_value > 0.05 else '⚠️  Residuals may not be normally distributed'}")
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(residuals, kde=True, bins=30, color='purple', alpha=0.7, ax=ax1)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_title("Residual Distribution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Residuals")
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot", fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20, color='teal')
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_title("Residuals vs Predicted", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Predicted Values")
    ax3.set_ylabel("Residuals")
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(y_test, y_pred, alpha=0.5, s=20, color='orange')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax4.set_title("Actual vs Predicted", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Actual Values")
    ax4.set_ylabel("Predicted Values")
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.savefig("experiments/results/experiment6_residual_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    metrics = {
        "Model": model_name,
        "Test RMSE": float(rmse),
        "Test MAE": float(mae),
        "Test R²": float(r2),
        "Residual Mean": float(np.mean(residuals)),
        "Residual Std": float(np.std(residuals)),
        "Residual Skewness": float(stats.skew(residuals)),
        "Residual Kurtosis": float(stats.kurtosis(residuals))
    }
    
    with open("experiments/results/experiment6_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    print("   MODEL TRAINING AGENT - COMPLETE PIPELINE")    
    try:
        X, y = load_and_prepare_data()
        
        best_name, best_model, scaler, X_train, X_test, y_train, y_test = \
            experiment_baseline_model_selection(X, y)
        
        tuned_model = experiment_hyperparameter_tuning(
            best_name, best_model, X_train, y_train, X_test, y_test
        )
        
        final_metrics = experiment_residual_analysis(tuned_model, X_test, y_test, best_name)
        
        # Final Summary
        print("\n" + "=" * 70)
        print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📁 Results saved in: experiments/results/")
        print("   - experiment4_model_comparison.png")
        print("   - experiment4_results.csv")
        print("   - experiment5_tuning_results.csv")
        print("   - experiment6_residual_analysis.png")
        print("   - experiment6_metrics.json")
        print("\n🎉 Pipeline completed!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()