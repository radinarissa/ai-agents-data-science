from agents.orchestration.orchestrator import OrchestratorAgent
from agents.data_processing.data_processing_agent import DataProcessingAgent
from agents.model_training.model_training_agent import ModelTrainingAgent
# noinspection PyUnresolvedReferences
from sklearn.ensemble import RandomForestRegressor
# noinspection PyUnresolvedReferences
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import os


class DataProcessingWrapper:
    def __init__(self):
        self.agent = None
        self.processed_df = None  # ✅ ADD: Store processed data

    def process(self, df):
        self.agent = DataProcessingAgent(df)
        features, validation = self.agent.run_pipeline(
            contamination=0.05,
            impute_strategy="median",
            feature_depth=1
        )
        # ✅ STORE the processed dataframe
        self.processed_df = features
        return features


def main():
    # ✅ FIX: Check if data file exists
    data_file = "data/raw/agentic_ai_performance_dataset_20250622.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"❌ Dataset not found: {data_file}\n"
                                f"Please ensure the file exists at the specified path.")

    print(f"✅ Found dataset: {data_file}")

    # Initialize orchestrator
    orchestrator = OrchestratorAgent()

    # Initialize data processor wrapper
    data_processor = DataProcessingWrapper()

    # Define experiments configuration
    experiments = [
        {
            'name': 'LinearRegression',
            'model': LinearRegression(),
            'params': {},
            'use_llm': False
        },
        {
            'name': 'Ridge',
            'model': Ridge(),
            'params': {'alpha': 1.0},
            'use_llm': False
        },
        {
            'name': 'RandomForest',
            'model': RandomForestRegressor(random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 10},
            'use_llm': True  # Will use Bayesian Optimization with LLM-suggested params
        }
    ]

    # Initialize model trainer
    model_trainer = ModelTrainingAgent(experiments=experiments, orchestrator=orchestrator)

    # Integrate agents
    orchestrator.integrate_agents(
        data_processor=data_processor,
        model_trainer=model_trainer
    )

    # Run the pipeline
    print("\n" + "=" * 60)
    print("Starting AI Agents Pipeline...")
    print("=" * 60 + "\n")

    results = orchestrator.run_pipeline(data_file)

    # ✅ FIX: Use PROCESSED data for visualizations, not raw data!
    # The data_processor now stores the processed dataframe
    if data_processor.processed_df is not None:
        processed_df = data_processor.processed_df
        print(
            f"\n✅ Using processed data for visualizations ({processed_df.shape[0]} rows, {processed_df.shape[1]} features)")
    else:
        # Fallback to raw data if processed data is not available
        print("\n⚠️ Warning: Processed data not available, using raw data")
        processed_df = pd.read_csv(data_file)

    # Generate visualizations using PROCESSED data
    charts, report = orchestrator.visualize_results(processed_df, results)

    # Generate final report
    orchestrator.generate_report(results)
    orchestrator.save_logs()

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"\n📊 View the report: results/report.html")
    print(f"📝 View the logs: results/pipeline_logs.txt\n")


if __name__ == "__main__":
    main()