from agents import OrchestratorAgent, DataProcessingAgent
import pandas as pd

def main():
    orchestrator = OrchestratorAgent()

    from agents.data_processing.main_agent import DataProcessingAgent
    data_processor = DataProcessingAgent(pd.DataFrame())
    orchestrator.integrate_agents(data_processor=data_processor)

    data_file = "data/raw/agentic_ai_performance_dataset_20250622.csv"
    results = orchestrator.run_pipeline(data_file)

    df = pd.read_csv(data_file)
    charts, report = orchestrator.visualize_results(df, results)

    orchestrator.generate_report(results)
    orchestrator.save_logs()

    print("\nPipeline completed! Open results/report.html to view the report.")


if __name__ == "__main__":
    main()