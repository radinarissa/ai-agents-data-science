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
            self.log("Data Processing Agent integrated")
        else:
            self.log("WARNING: Data Processing Agent not provided")
            self.log("System will require Data Processing Agent to run")

        if model_trainer:
            self.model_trainer = model_trainer
            self.log("Model Training Agent integrated")

            if hasattr(model_trainer, 'llm_enabled') and model_trainer.llm_enabled:
                self.log("LLM capabilities detected in Model Training Agent")
                self.metrics['llm_enabled'] = True
            else:
                self.metrics['llm_enabled'] = False
        else:
            self.log("WARNING: Model Training Agent not provided")
            self.log("System will require Model Training Agent to run")

        return self

    def load_data(self, data_path):
        self.log(f"Loading data from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            df = pd.read_csv(data_path)
            self.log(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            self.metrics['original_rows'] = df.shape[0]
            self.metrics['original_columns'] = df.shape[1]

            return df

        except Exception as e:
            self.log(f"ERROR loading data: {str(e)}")
            raise

    def process_data(self, df):
        self.log("=" * 60)
        self.log("STEP 1: DATA PROCESSING")
        self.log("=" * 60)

        start_time = time.time()

        try:
            if not self.data_processor:
                self.log("ERROR: Data Processing Agent is required!")
                raise ValueError("Data Processing Agent must be provided via integrate_agents()")

            processed_df = self.data_processor.process(df)
            self.log("Data processed by Data Processing Agent")

            processing_time = time.time() - start_time
            self.metrics['processing_time'] = round(processing_time, 2)
            self.metrics['processed_rows'] = processed_df.shape[0]
            self.metrics['processed_columns'] = processed_df.shape[1]

            rows_removed = self.metrics['original_rows'] - processed_df.shape[0]
            cols_added = processed_df.shape[1] - self.metrics['original_columns']

            self.log(f"Processing completed in {processing_time:.2f} seconds")
            self.log(f"Rows removed: {rows_removed}")
            self.log(f"Columns added (features): {cols_added}")

            return processed_df

        except Exception as e:
            self.log(f"ERROR in data processing: {str(e)}")
            raise

    def train_model(self, processed_df):
        self.log("=" * 60)
        self.log("STEP 2: MODEL TRAINING")
        self.log("=" * 60)

        start_time = time.time()

        try:
            if not self.model_trainer:
                self.log("ERROR: Model Training Agent is required!")
                raise ValueError("Model Training Agent must be provided via integrate_agents()")

            results = self.model_trainer.train(processed_df)
            self.log("Model trained by Model Training Agent")

            training_time = time.time() - start_time
            self.metrics['training_time'] = round(training_time, 2)

            if isinstance(results, dict):
                if 'model_name' in results:
                    self.log(f"Selected model: {results['model_name']}")
                if 'score' in results:
                    self.log(f"Model score: {results['score']:.4f}")
                if 'best_params' in results:
                    self.log(f"Best parameters: {results['best_params']}")

            self.log(f"Training completed in {training_time:.2f} seconds")

            return results

        except Exception as e:
            self.log(f"ERROR in model training: {str(e)}")
            raise

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
            self.log("PIPELINE COMPLETED SUCCESSFULLY")
            self.log(f"Total execution time: {pipeline_time:.2f} seconds")
            self.log("=" * 60)

            return {
                'status': 'success',
                'results': results,
                'metrics': self.metrics,
                'logs': self.pipeline_logs
            }

        except Exception as e:
            pipeline_time = time.time() - pipeline_start
            self.metrics['total_pipeline_time'] = round(pipeline_time, 2)

            self.log("=" * 60)
            self.log("PIPELINE FAILED")
            self.log(f"Error: {str(e)}")
            self.log(f"Failed after: {pipeline_time:.2f} seconds")
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
            self.log("\nMETRICS:")
            for key, value in results['metrics'].items():
                self.log(f"  {key}: {value}")

            self.log("\nStatus: SUCCESS")

            if 'results' in results and isinstance(results['results'], dict):
                self.log("\nMODEL RESULTS:")
                for key, value in results['results'].items():
                    if key not in ['model', 'predictions']:
                        self.log(f"  {key}: {value}")
        else:
            self.log("\nStatus: FAILED")
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

        self.log("=" * 60)
        self.log("GENERATING VISUALIZATIONS")
        self.log("=" * 60)

        try:
            visualizer = VisualizerAgent()

            charts = {}
            charts['Data Overview'] = visualizer.plot_data_overview(df)
            self.log("Data overview chart generated")

            charts['Pipeline Metrics'] = visualizer.plot_metrics(self.metrics)
            self.log("Pipeline metrics chart generated")

            charts['Processing Time'] = visualizer.plot_processing_time(self.metrics)
            self.log("Processing time chart generated")

            report_gen = ReportGenerator()
            report_path = report_gen.generate_html_report(results, charts)

            self.log(f"HTML report generated: {report_path}")
            self.log("=" * 60)

            return charts, report_path

        except Exception as e:
            self.log(f"ERROR generating visualizations: {str(e)}")
            return {}, None

    def log_llm_interaction(self, prompt, response, tokens_used=None, model_name=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        response_preview = response[:100] + "..." if len(response) > 100 else response

        log_entry = f"""
[{timestamp}] LLM INTERACTION
  Model: {model_name if model_name else 'N/A'}
  Prompt: {prompt_preview}
  Response: {response_preview}
  Tokens: {tokens_used if tokens_used else 'N/A'}
"""
        self.pipeline_logs.append(log_entry)
        print(log_entry)

        if 'llm_interactions' not in self.metrics:
            self.metrics['llm_interactions'] = 0
        self.metrics['llm_interactions'] += 1

        if tokens_used:
            if 'total_tokens_used' not in self.metrics:
                self.metrics['total_tokens_used'] = 0
            self.metrics['total_tokens_used'] += tokens_used

    def log_model_selection(self, selected_model, candidates, scores):
        self.log("=" * 60)
        self.log("MODEL SELECTION PROCESS")
        self.log(f"Candidates tested: {len(candidates)}")
        self.log("")

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for i, (model_name, score) in enumerate(sorted_scores, 1):
            marker = "[SELECTED]" if model_name == selected_model else ""
            self.log(f"  {i}. {model_name}: {score:.4f} {marker}")

        self.log("")
        self.log(f"Selected model: {selected_model}")
        self.log("=" * 60)

        self.metrics['selected_model'] = selected_model
        self.metrics['best_score'] = scores[selected_model]
        self.metrics['models_tested'] = len(candidates)

    def log_hyperparameter_tuning(self, model_name, best_params, tuning_method, iterations):
        self.log("=" * 60)
        self.log("HYPERPARAMETER TUNING")
        self.log(f"Model: {model_name}")
        self.log(f"Method: {tuning_method}")
        self.log(f"Iterations: {iterations}")
        self.log(f"Best parameters:")

        for param, value in best_params.items():
            self.log(f"  {param}: {value}")

        self.log("=" * 60)

        self.metrics['tuning_method'] = tuning_method
        self.metrics['tuning_iterations'] = iterations