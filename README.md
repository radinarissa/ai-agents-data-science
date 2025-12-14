# AI Agents for Data Science Automation

Multi-agent система за автоматизация на ML pipeline – от обработка на данни до обучение, оценка и визуализация.

## 🎯 Какво прави?

Три специализирани агента работят заедно за автоматизиране на data science процеси:
- **Data Processing Agent**: Cleaning, feature engineering, validation (Experiments 7–9)
- **Model Training Agent**: Model selection, hyperparameter tuning, evaluation (Experiments 4–6)
- **Orchestrator Agent**: Координация, визуализации, отчети (Experiments 1–3)

Допълнително:
- **Experiment 10**: Визуализация на обработените данни (диаграми, корелации, статистики)

---

## 🧪 Experiments Overview

| Experiment | Agent            | Purpose                                | Output Files |
|------------|------------------|----------------------------------------|--------------|
| 1–3        | Orchestrator     | Initialize pipeline, logging, reporting| results/report.html, results/pipeline_logs.txt |
| 7–9        | Data Processing  | Cleaning, feature engineering, validation | data/processed/processed_dataset.csv, experiments/validation_report.csv |
| 4–6        | Model Training   | Model selection, hyperparameter tuning, residual analysis | experiments/results/all_results_4_5_6.json, residual plots |
| 10         | Visualization    | Histograms, correlation matrix, descriptive stats | experiments/results/histograms.png, correlation_matrix.png, descriptive_statistics.csv |

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/radinarissa/ai-agents-data-science.git
cd ai-agents-data-science
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
2. Download Dataset
Download от Kaggle → Сложи agentic_ai_performance_dataset_20250622.csv в data/raw/

3. Run Pipeline
bash
# Full pipeline (Experiments 1–3, 7–9, 4–6)
python main.py

# Advanced experiments only (Experiments 4–6)
python run_experiments.py

# Dataset visualization (Experiment 10)
python experiment_10.py
📁 Key Files
Code
├── main.py                          # Full pipeline execution (1–3, 7–9, 4–6)
├── run_experiments.py               # Experiments 4–6 (model selection, tuning, analysis)
├── experiment_10.py                 # Experiment 10 (dataset visualization)
├── agents/
│   ├── data_processing/
│   │   ├── data_processing_agent.py # Experiments 7–9
│   │   ├── data_cleaning.py         # Outlier detection
│   │   ├── feature_engineering.py   # DFS feature generation
│   │   └── data_validation.py       # Feature filtering
│   ├── model_training/
│   │   ├── model_training_agent.py  # Training coordinator
│   │   ├── agent_model_selection.py # Experiment 4
│   │   ├── agent_hyperparameter_tuning.py # Experiment 5
│   │   └── model_evaluation.py      # Experiment 6
│   └── orchestration/
│       ├── orchestrator.py          # Experiments 1–3
│       ├── visualizer.py            # Charts generation
│       └── report_generator.py      # HTML reports
└── data/
    ├── raw/                         # Original dataset
    └── processed/                   # Processed data (auto-generated)
🧪 Testing
Test 1: Basic Pipeline
bash
python main.py
Очаквано:

Data processing: ~2 seconds

Model training: ~30 seconds

Output: results/report.html, results/pipeline_logs.txt

Metrics: R² ≈ 0.91, RMSE ≈ 0.044

Test 2: Advanced Experiments
bash
python run_experiments.py
Очаквано:

Experiment 4: Model Selection (~17 sec) → RandomForest wins

Experiment 5: Hyperparameter Tuning (~62 sec) → Optimized params

Experiment 6: Residual Analysis (~3 sec) → 4 plots + metrics

Output: experiments/results/all_results_4_5_6.json, plots

Test 3: Dataset Visualization
bash
python experiment_10.py
Очаквано:

Histograms of numeric features

Correlation matrix heatmap

Descriptive statistics CSV

Output: experiments/results/histograms.png, correlation_matrix.png, descriptive_statistics.csv

📊 Results
След run, провери:

Console Output:

Data processing logs

Model metrics (R², RMSE, MAE)

Training time

Files:

Code
results/
├── report.html              # HTML dashboard
├── pipeline_logs.txt        # Full execution log
├── actual_vs_predicted.png
└── residual_distribution.png

experiments/results/
├── all_results_4_5_6.json   # Experiment 4–6 metrics
├── experiment6_*.png        # Residual analysis plots
├── histograms.png           # Experiment 10 histograms
├── correlation_matrix.png   # Experiment 10 correlation heatmap
└── descriptive_statistics.csv
Expected Metrics (after data leakage fix):

R² Score: ~0.906

RMSE: ~0.044

MAE: ~0.032

Dataset: 5000 rows, 44 features (after cleaning)

⚙️ Configuration
Edit в main.py:

python
# Data processing
contamination=0.05     # Outlier detection threshold
impute_strategy="median"
feature_depth=1        # DFS depth

# Model training
experiments = [...]    # Add/remove models

📈 Key Features
✅ Data Leakage Prevention: Target separation преди feature engineering ✅ Optimized Grid Search: 4 комбинации (8.7x по-бързо) ✅ Automated Feature Engineering: DFS с featuretools ✅ Model Comparison: 4 models (RandomForest, Ridge, LinearReg, SVR) ✅ Residual Analysis: Shapiro-Wilk test, Q-Q plot, distribution ✅ Dataset Visualization: Histograms, correlation matrix, descriptive stats (Experiment 10) ✅ Comprehensive Logging: Orchestrator tracks всичко

🎓 Project Info
Dataset: Agentic AI Performance Dataset Tech Stack: Python, scikit-learn, featuretools, pandas, matplotlib, seaborn, statsmodels

Quick Test:

bash
python main.py
→ трябва да завърши за ~30 секунди с R² ≈ 0.91