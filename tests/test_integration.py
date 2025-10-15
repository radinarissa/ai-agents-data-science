import unittest
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import OrchestratorAgent, VisualizerAgent, ReportGenerator


class TestOrchestratorAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_path = "data/raw/agentic_ai_performance_dataset_20250622.csv"
        if not os.path.exists(cls.test_data_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cls.test_data_path = os.path.join(project_root, "data", "raw",
                                              "agentic_ai_performance_dataset_20250622.csv")

    def setUp(self):
        self.orchestrator = OrchestratorAgent()

    def test_orchestrator_initialization(self):
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNone(self.orchestrator.data_processor)
        self.assertIsNone(self.orchestrator.model_trainer)
        self.assertEqual(len(self.orchestrator.pipeline_logs), 0)
        self.assertEqual(len(self.orchestrator.metrics), 0)

    def test_load_data_success(self):
        df = self.orchestrator.load_data(self.test_data_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertGreater(len(df.columns), 0)
        self.assertIn('original_rows', self.orchestrator.metrics)
        self.assertIn('original_columns', self.orchestrator.metrics)

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_data("nonexistent_file.csv")

    def test_process_data_default(self):
        df = self.orchestrator.load_data(self.test_data_path)
        processed_df = self.orchestrator.process_data(df)

        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertIn('processing_time', self.orchestrator.metrics)
        self.assertIn('processed_rows', self.orchestrator.metrics)
        self.assertGreaterEqual(self.orchestrator.metrics['processing_time'], 0)

    def test_train_model_default(self):
        df = self.orchestrator.load_data(self.test_data_path)
        processed_df = self.orchestrator.process_data(df)
        results = self.orchestrator.train_model(processed_df)

        self.assertIsInstance(results, dict)
        self.assertIn('training_time', self.orchestrator.metrics)
        self.assertGreaterEqual(self.orchestrator.metrics['training_time'], 0)

    def test_run_pipeline_success(self):
        results = self.orchestrator.run_pipeline(self.test_data_path)

        self.assertIsInstance(results, dict)
        self.assertEqual(results['status'], 'success')
        self.assertIn('metrics', results)
        self.assertIn('logs', results)
        self.assertIn('results', results)
        self.assertIn('total_pipeline_time', results['metrics'])

    def test_run_pipeline_invalid_path(self):
        results = self.orchestrator.run_pipeline("invalid_path.csv")

        self.assertEqual(results['status'], 'error')
        self.assertIn('error', results)

    def test_logging_functionality(self):
        initial_log_count = len(self.orchestrator.pipeline_logs)

        self.orchestrator.log("Test message")

        self.assertEqual(len(self.orchestrator.pipeline_logs), initial_log_count + 1)
        self.assertIn("Test message", self.orchestrator.pipeline_logs[-1])

    def test_save_logs(self):
        self.orchestrator.log("Test log entry")

        test_log_path = "results/test_logs.txt"
        self.orchestrator.save_logs(filepath=test_log_path)

        self.assertTrue(os.path.exists(test_log_path))

        with open(test_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Test log entry", content)

        os.remove(test_log_path)


class TestVisualizerAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_path = "data/raw/agentic_ai_performance_dataset_20250622.csv"
        if not os.path.exists(cls.test_data_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cls.test_data_path = os.path.join(project_root, "data", "raw",
                                              "agentic_ai_performance_dataset_20250622.csv")

        cls.df = pd.read_csv(cls.test_data_path)
        cls.test_output_dir = "results/test_figures"

    def setUp(self):
        self.visualizer = VisualizerAgent(output_dir=self.test_output_dir)

    def test_visualizer_initialization(self):
        self.assertIsNotNone(self.visualizer)
        self.assertTrue(os.path.exists(self.test_output_dir))

    def test_plot_data_overview(self):
        filepath = self.visualizer.plot_data_overview(self.df, filename="test_overview.png")

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.png'))

    def test_plot_metrics(self):
        test_metrics = {
            'metric1': 100,
            'metric2': 200,
            'metric3': 150
        }

        filepath = self.visualizer.plot_metrics(test_metrics, filename="test_metrics.png")

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.png'))

    def test_plot_processing_time(self):
        test_metrics = {
            'processing_time': 0.5,
            'training_time': 1.2,
            'total_pipeline_time': 1.7
        }

        filepath = self.visualizer.plot_processing_time(test_metrics, filename="test_time.png")

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.png'))


class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.report_gen = ReportGenerator(output_dir="results/test_reports")

    def test_report_generator_initialization(self):
        self.assertIsNotNone(self.report_gen)
        self.assertTrue(os.path.exists("results/test_reports"))

    def test_generate_html_report(self):
        test_results = {
            'status': 'success',
            'metrics': {
                'original_rows': 5000,
                'original_columns': 26,
                'processing_time': 0.01
            }
        }

        test_charts = {
            'Test Chart': 'results/figures/test.png'
        }

        filepath = self.report_gen.generate_html_report(
            test_results,
            test_charts,
            filename="test_report.html"
        )

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.html'))

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('success', content.lower())
            self.assertIn('5000', content)


class TestFullIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_path = "data/raw/agentic_ai_performance_dataset_20250622.csv"
        if not os.path.exists(cls.test_data_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cls.test_data_path = os.path.join(project_root, "data", "raw",
                                              "agentic_ai_performance_dataset_20250622.csv")

    def test_full_pipeline_with_visualization(self):
        orchestrator = OrchestratorAgent()

        results = orchestrator.run_pipeline(self.test_data_path)

        self.assertEqual(results['status'], 'success')

        df = pd.read_csv(self.test_data_path)
        charts, report = orchestrator.visualize_results(df, results)

        self.assertIsInstance(charts, dict)
        self.assertGreater(len(charts), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)