from agents import OrchestratorAgent


def main():
    orchestrator = OrchestratorAgent()

    data_file = "data/raw/agentic_ai_performance_dataset_20250622.csv"

    results = orchestrator.run_pipeline(data_file)
    print(f"\nPipeline results: {results}")


if __name__ == "__main__":
    main()