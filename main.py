from agents import OrchestratorAgent


def main():
    orchestrator = OrchestratorAgent()

    data_file = "data/raw/agentic_ai_performance_dataset_20250622.csv"

    results = orchestrator.run_pipeline(data_file)

    df = orchestrator.load_data(data_file)
    charts, report = orchestrator.visualize_results(df, results)

    orchestrator.generate_report(results)

    orchestrator.save_logs()

    print("\n Pipeline completed! Open results/report.html to view the report.")


if __name__ == "__main__":
    main()