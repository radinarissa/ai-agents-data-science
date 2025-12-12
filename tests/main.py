import sys, os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.data_processing.data_processing_agent import DataProcessingAgent
from agents.model_training.model_training_agent import ModelTrainingAgent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("🚀 FULL PIPELINE – Data Processing + Model Training")

    # Step 1: Load raw dataset
    raw_df = pd.read_csv("data/raw/agentic_ai_performance_dataset_20250622.csv")

    # Step 2: Run DataProcessingAgent
    print("\n🔧 Running DataProcessingAgent...")
    dp_agent = DataProcessingAgent(raw_df)
    processed_df, validation_report = dp_agent.run_pipeline()

    # Step 3: Extract features and target
    X = processed_df.drop(columns=["success_rate"])
    y = processed_df["success_rate"]

    # Step 4: Train/test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Define experiments
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    experiments = [
        {"name": "LinearRegression", "model": LinearRegression(), "params": {}, "use_llm": False},
        {"name": "Ridge", "model": Ridge(), "params": {"alpha": 1.0}, "use_llm": False},
        {"name": "RandomForest", "model": RandomForestRegressor(), "params": {"n_estimators": 100}, "use_llm": False},
        {"name": "SVR", "model": SVR(), "params": {"C": 1.0}, "use_llm": False}
    ]

    # Step 6: Run ModelTrainingAgent
    print("\n🤖 Running ModelTrainingAgent...")
    trainer = ModelTrainingAgent(experiments)
    best_model_info = trainer.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

    print(f"\n🏆 Best model: {best_model_info['name']}")
    print(f"   R² Score: {best_model_info['R2 Score']:.4f}")
    print(f"   RMSE: {best_model_info['RMSE']:.4f}")

    print("\n✅ Pipeline complete.")

if __name__ == "__main__":
    main()
