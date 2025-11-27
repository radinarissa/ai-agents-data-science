# AI Agents for Data Science Automation

Multi-agent система с три специализирани модула за автоматизация на data science процеси.

##  Modules

###  Data Processing Agent
**Отговорности:**
- Data cleaning (missing values, outliers)
- Feature engineering
- Data validation

**Използване:**
```python
from agents import DataProcessingAgent

processor = DataProcessingAgent()
cleaned_data = processor.process(raw_data)
```

**Dependencies:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

### Model Training Agent
**Отговорности:**
- Automated model selection
- Hyperparameter tuning
- Model evaluation

**Използване:**
```python
from agents import ModelTrainingAgent

trainer = ModelTrainingAgent()
results = trainer.train(processed_data)
```

**Dependencies:**
```
scikit-learn>=1.3.0
optuna>=3.5.0
xgboost>=2.0.0
```

---

### Orchestrator Agent
**Отговорности:**
- Pipeline coordination
- Agent integration
- Visualization & reporting

**Използване:**
```python
from agents import OrchestratorAgent

orchestrator = OrchestratorAgent()
orchestrator.integrate_agents(data_processor, model_trainer)
results = orchestrator.run_pipeline("data/raw/dataset.csv")
```

**Dependencies:**
```
matplotlib>=3.7.0
seaborn>=0.12.0
langchain>=0.1.0
```

## Setup

```bash
git clone https://github.com/radinarissa/ai-agents-data-science.git
cd ai-agents-data-science
pip install -r requirements.txt
```

**All Dependencies:**
```
# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Model Training
optuna>=3.5.0
xgboost>=2.0.0

# Orchestration & Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
langchain>=0.1.0

# Utilities
jupyter>=1.0.0
```

##  Usage

**Quick Start:**
```bash
python main.py
```

**Run Experiments:**
```bash
python run_experiments.py
```

**Full Pipeline:**
```python
from agents import OrchestratorAgent, DataProcessingAgent, ModelTrainingAgent

# Initialize agents
processor = DataProcessingAgent()
trainer = ModelTrainingAgent()
orchestrator = OrchestratorAgent()

# Integrate and run
orchestrator.integrate_agents(processor, trainer)
results = orchestrator.run_pipeline("data.csv")
```

## Структура на проекта

```
ai-agents-data-science/
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py          # Orchestrator Agent
│   ├── visualizer.py            # Visualization Agent
│   └── report_generator.py      # Report Generator Agent
│
├── data/
│   ├── raw/
│   │   └── agentic_ai_performance_dataset_20250622.csv
│   └── processed/               # Temporary processed data
│
├── results/
│   ├── figures/                 # Generated charts
│   ├── experiments/             # Experiment results
│   ├── report.html             # HTML report
│   └── pipeline_logs.txt       # Execution logs
│
├── main.py                      # Main entry point
├── run_experiments.py           # Experiment runner
├── data_analysis.ipynb          # Jupyter notebook analysis
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## Dataset

**Source:** [Kaggle - Agentic AI Performance Dataset](https://www.kaggle.com/datasets/bismasajjad/agentic-ai-performance-and-capabilities-dataset)

- 5000 records, 26 features
- AI agents performance metrics

**Key Features:**
- `success_rate`, `accuracy_score`, `efficiency_score`
- `memory_usage_mb`, `cpu_usage_percent`
- `autonomy_level`, `task_complexity`

## Experiments

**Benchmarks:**
- Baseline pipeline
- Performance iterations (3-5 runs)
- Data size scaling (100-5000 records)

**Results:** `results/experiments/experiment_results.json`
