import pandas as pd
import time
import json
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestration  import OrchestratorAgent
from agents.data_processing.data_processing_agent import DataProcessingAgent


class ExperimentRunner:
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = []
        self.output_dir = "results/experiments"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_orchestrator(self):
        orchestrator = OrchestratorAgent()

        data_processor = DataProcessingAgent(pd.DataFrame())

        orchestrator.integrate_agents(data_processor=data_processor)
        return orchestrator

    def run_experiment(self, name, config):
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {name}")
        print(f"{'=' * 60}")

        start_time = time.time()

        orchestrator = self.create_orchestrator()
        pipeline_results = orchestrator.run_pipeline(self.data_path)

        execution_time = time.time() - start_time

        experiment_result = {
            'name': name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': config,
            'execution_time': round(execution_time, 3),
            'status': pipeline_results['status'],
            'metrics': pipeline_results['metrics']
        }

        if pipeline_results['status'] == 'error':
            experiment_result['error'] = pipeline_results['error']

        self.results.append(experiment_result)

        print(f"Experiment completed in {execution_time:.2f}s")

        return experiment_result

    def run_multiple_iterations(self, iterations=5):
        print(f"\n{'=' * 60}")
        print(f"Running benchmark: {iterations} iterations")
        print(f"{'=' * 60}")

        times = []

        for i in range(iterations):
            print(f"\nIteration {i + 1}/{iterations}")

            start = time.time()
            orchestrator = self.create_orchestrator()
            orchestrator.run_pipeline(self.data_path)
            elapsed = time.time() - start

            times.append(elapsed)
            print(f"Time: {elapsed:.3f}s")

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        benchmark_result = {
            'name': 'Performance Benchmark',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iterations': iterations,
            'times': times,
            'avg_time': round(avg_time, 3),
            'min_time': round(min_time, 3),
            'max_time': round(max_time, 3)
        }

        self.results.append(benchmark_result)

        print(f"\n{'=' * 60}")
        print("Benchmark Results:")
        print(f"Average: {avg_time:.3f}s")
        print(f"Min: {min_time:.3f}s")
        print(f"Max: {max_time:.3f}s")
        print(f"{'=' * 60}")

        return benchmark_result

    def compare_data_sizes(self, sample_sizes=[100, 500, 1000, 2500, 5000]):
        print(f"\n{'=' * 60}")
        print("Comparing performance across data sizes")
        print(f"{'=' * 60}")

        df = pd.read_csv(self.data_path)
        comparison_results = []

        for size in sample_sizes:
            if size > len(df):
                print(f"Skipping size {size} (larger than dataset)")
                continue

            print(f"\nTesting with {size} rows...")

            sample_df = df.sample(n=size, random_state=42)
            temp_path = f"data/processed/sample_{size}.csv"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            sample_df.to_csv(temp_path, index=False)

            start = time.time()
            orchestrator = self.create_orchestrator()
            result = orchestrator.run_pipeline(temp_path)
            elapsed = time.time() - start

            comparison_results.append({
                'size': size,
                'time': round(elapsed, 3),
                'status': result['status']
            })

            print(f"Size {size}: {elapsed:.3f}s")

            os.remove(temp_path)

        size_comparison = {
            'name': 'Data Size Comparison',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': comparison_results
        }

        self.results.append(size_comparison)

        return size_comparison

    def save_results(self, filename="experiment_results.json"):
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {filepath}")

        return filepath

    def generate_summary(self):
        print(f"\n{'=' * 60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total experiments: {len(self.results)}")

        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result['name']}")
            if 'execution_time' in result:
                print(f"   Time: {result['execution_time']}s")
            if 'avg_time' in result:
                print(f"   Avg time: {result['avg_time']}s")
            if 'status' in result:
                print(f"   Status: {result['status']}")

        print(f"{'=' * 60}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, "data", "raw", "agentic_ai_performance_dataset_20250622.csv")

    runner = ExperimentRunner(data_file)

    runner.run_experiment(
        name="Baseline Pipeline",
        config={"type": "default"}
    )

    runner.run_multiple_iterations(iterations=3)

    runner.compare_data_sizes(sample_sizes=[100, 500, 1000, 2500])

    runner.save_results()

    runner.generate_summary()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()