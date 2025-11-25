import pandas as pd
import os
import time
from datetime import datetime


class OrchestratorAgent:
    def __init__(self):
        self.data_processor = None
        self.model_trainer = None
        self.pipeline_logs = []
        self.metrics = {}

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.pipeline_logs.append(log_entry)
        print(log_entry)

    def integrate_agents(self, data_processor=None, model_trainer=None):
        self.log("Integrating agents...")

        if data_processor:
            self.data_processor = data_processor
            self.log("Data Processor integrated")
        else:
            self.log("Data Processor not provided - will use default processing")

        if model_trainer:
            self.model_trainer = model_trainer
            self.log("Model Trainer integrated")
        else:
            self.log("Model Trainer not provided - will skip training")

        return self

    def load_data(self, data_path):
        self.log(f"Loading data from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            df = pd.read_csv(data_path)
            self.log(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            self.metrics['original_rows'] = df.shape[0]
            self.metrics['original_columns'] = df.shape[1]

            return df
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            raise

    def process_data(self, df):
        self.log("Starting data processing...")
        start_time = time.time()

        try:
            if self.data_processor:
                processed_df = self.data_processor.process(df)
                self.log(f"Data processed by Data Processor Agent")
            else:
                processed_df = self._default_processing(df)
                self.log(f"Data processed with default method")

            processing_time = time.time() - start_time
            self.metrics['processing_time'] = round(processing_time, 2)
            self.metrics['processed_rows'] = processed_df.shape[0]

            self.log(f"Processing completed in {processing_time:.2f} seconds")
            return processed_df

        except Exception as e:
            self.log(f"Error in data processing: {str(e)}")
            raise

    def _default_processing(self, df):
        self.log("Applying default processing (removing nulls)")
        original_rows = len(df)
        df_clean = df.dropna()
        removed_rows = original_rows - len(df_clean)

        if removed_rows > 0:
            self.log(f"Removed {removed_rows} rows with missing values")

        return df_clean

    def train_model(self, processed_df):
        self.log("Starting model training...")
        start_time = time.time()

        try:
            if self.model_trainer:
                results = self.model_trainer.train(processed_df)
                self.log(f"Model trained by Model Trainer Agent")
            else:
                results = self._default_training(processed_df)
                self.log(f"Basic analysis completed (no model training)")

            training_time = time.time() - start_time
            self.metrics['training_time'] = round(training_time, 2)

            self.log(f"Training completed in {training_time:.2f} seconds")
            return results

        except Exception as e:
            self.log(f"Error in model training: {str(e)}")
            raise

    def _default_training(self, df):
        self.log("Generating basic statistics")

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        stats = {
            'numeric_columns': len(numeric_cols),
            'summary': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
        }

        return stats

    def run_pipeline(self, data_path):
        self.log("=" * 60)
        self.log("STARTING AI AGENTS PIPELINE")
        self.log("=" * 60)

        pipeline_start = time.time()

        try:
            df = self.load_data(data_path)

            processed_df = self.process_data(df)

            results = self.train_model(processed_df)

            pipeline_time = time.time() - pipeline_start
            self.metrics['total_pipeline_time'] = round(pipeline_time, 2)

            self.log("=" * 60)
            self.log(f"PIPELINE COMPLETED SUCCESSFULLY")
            self.log(f"Total time: {pipeline_time:.2f} seconds")
            self.log("=" * 60)

            return {
                'status': 'success',
                'results': results,
                'metrics': self.metrics,
                'logs': self.pipeline_logs
            }

        except Exception as e:
            self.log("=" * 60)
            self.log(f"PIPELINE FAILED: {str(e)}")
            self.log("=" * 60)

            return {
                'status': 'error',
                'error': str(e),
                'metrics': self.metrics,
                'logs': self.pipeline_logs
            }

    def generate_report(self, results):
        self.log("\n" + "=" * 60)
        self.log("PIPELINE REPORT")
        self.log("=" * 60)

        if results['status'] == 'success':
            self.log("\n METRICS:")
            for key, value in results['metrics'].items():
                self.log(f"  {key}: {value}")

            self.log("\n Status: SUCCESS")
        else:
            self.log(f"\n Status: FAILED")
            self.log(f"Error: {results['error']}")

        self.log("=" * 60)

        return results

    def save_logs(self, filepath="results/pipeline_logs.txt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.pipeline_logs))

        self.log(f"Logs saved to: {filepath}")

    def visualize_results(self, df, results):
        from .visualizer import VisualizerAgent
        from .report_generator import ReportGenerator

        self.log("Generating visualizations...")

        try:
            visualizer = VisualizerAgent()

            charts = {}
            charts['Data Overview'] = visualizer.plot_data_overview(df)
            charts['Pipeline Metrics'] = visualizer.plot_metrics(self.metrics)
            charts['Processing Time'] = visualizer.plot_processing_time(self.metrics)

            self.log(f"Charts generated")

            report_gen = ReportGenerator()
            report_path = report_gen.generate_html_report(results, charts)

            self.log(f"HTML report: {report_path}")

            return charts, report_path

        except Exception as e:
            self.log(f"Error generating visualizations: {str(e)}")
            return {}, None