import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

from main_agent import DataProcessingAgent


if __name__ == "__main__":
    df = pd.read_csv("data/raw/agentic_ai_performance_dataset_20250622.csv")
    
    df = pd.read_csv("data/raw/agentic_ai_performance_dataset_20250622.csv")
    
    agent = DataProcessingAgent(df)
    features, validation = agent.run_pipeline(
        contamination=0.05,
        impute_strategy="median",
        feature_depth=1
    )

    print(validation.head())

