from agents import OrchestratorAgent

def main():
    orchestrator = OrchestratorAgent()

    data_file = "data/raw/agentic_ai_performance_dataset_20250622.csv"

    results = orchestrator.run_pipeline(data_file)

    orchestrator.generate_report(results)

    orchestrator.save_logs()

if __name__ == "__main__":
    main()