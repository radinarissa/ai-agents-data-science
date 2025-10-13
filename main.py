from agents import OrchestratorAgent

def main():
    orchestrator = OrchestratorAgent()
    results = orchestrator.run_pipeline("data/raw/sample.csv")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()