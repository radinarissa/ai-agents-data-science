import pandas as pd
import os


class OrchestratorAgent:
    def __init__(self):
        self.data_processor = None
        self.model_trainer = None

    def run_pipeline(self, data_path):
        print(f"Starting pipeline: {data_path}")

        if not os.path.exists(data_path):
            print(f"Error: File {data_path} not found!")
            return {'status': 'error', 'message': 'File not found'}

        df = pd.read_csv(data_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:\n{df.head()}")

        results = {
            'status': 'success',
            'rows': df.shape[0],
            'columns': df.shape[1],
            'data_path': data_path
        }

        return results

    def integrate_agents(self, data_processor, model_trainer):
        self.data_processor = data_processor
        self.model_trainer = model_trainer

    def generate_report(self, results):
        print("Generating report...")
        return results