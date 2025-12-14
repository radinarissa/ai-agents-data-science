# AI Agents for Data Science Automation

Multi-agent система за автоматизация на ML pipeline - от обработка на данни до обучение и оценка на модели.

## 🎯 Какво прави?

Три специализирани агента работят заедно за автоматизиране на data science процеси: 

 **Data Processing Agent**: Cleaning, feature engineering, validation (Experiments 7–9) 

 **Model Training Agent**: Model selection, hyperparameter tuning, evaluation (Experiments 4–6)

 **Orchestrator Agent**: Координация, визуализации, отчети (Experiments 1–3) 
 
Допълнително: 

**Experiment 10**: Визуализация на обработените данни (диаграми, корелации, статистики) 
 
 ## 🧪 Experiments Overview 
| Experiment | Agent          | Purpose                                   | Output Files |
|------------|----------------|-------------------------------------------|--------------|
| 1–3        | Orchestrator   | Initialize pipeline, logging, reporting    | `results/report.html`, `results/pipeline_logs.txt` |
| 7–9        | Data Processing| Cleaning, feature engineering, validation | `data/processed/processed_dataset.csv`, `experiments/validation_report.csv` |
| 4–6        | Model Training | Model selection, hyperparameter tuning, residual analysis | `experiments/results/all_results_4_5_6.json`, residual plots |
| 10         | Visualization  | Histograms, correlation matrix, descriptive stats | `experiments/results/histograms.png`, `correlation_matrix.png`, `descriptive_statistics.csv` |

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/radinarissa/ai-agents-data-science.git
cd ai-agents-data-science
pip install -r requirements.txt
```

### 2. Dataset

Downloaed from [Kaggle](https://www.kaggle.com/datasets/bismasajjad/agentic-ai-performance-and-capabilities-dataset)  
→ `agentic_ai_performance_dataset_20250622.csv` в `data/raw/`

### 3. Run Pipeline

```bash
# Full pipeline (data processing + model training)
python main.py

# Advanced experiments (model selection, tuning, residual analysis)
python run_experiments.py

# Interactive EDA (Jupyter Notebook)
jupyter notebook notebooks/data_analysis.ipynb
```

## 📁 Project Structure

```
├── main.py                          # Full pipeline execution
├── run_experiments.py               # Experiments 4-6
├── agents/
│   ├── data_processing/             # Data cleaning & feature engineering
│   ├── model_training/              # Model selection, tuning, evaluation
│   └── orchestration/               # Agent coordination & logging
├── data/
│   ├── raw/                         # Original dataset (put here)
│   └── processed/                   # Auto-generated processed data
├── notebooks/
│   ├── data_analysis.ipynb          # Interactive EDA
│   └── results/                     # Notebook outputs (charts, HTML)
├── experiments/results/             # Experiment outputs
│   ├── all_results_4_5_6.json
│   └── experiment6_*.png (4 plots)
└── results/
    ├── report.html                  # Main pipeline report
    └── pipeline_logs.txt
```

## 🧪 Testing

### Test 1: Basic Pipeline
```bash
python main.py
```
**Очаквано:**
- Time: ~30 seconds
- Output: `results/report.html`
- Metrics: R² ≈ 0.91, RMSE ≈ 0.044

### Test 2: Advanced Experiments
```bash
python run_experiments.py
```
**Очаквано:**
- Experiment 4: Model Selection (~17 sec)
- Experiment 5: Hyperparameter Tuning (~62 sec, OPTIMIZED)
- Experiment 6: Residual Analysis (~3 sec, 4 plots)
- Output: `experiments/results/all_results_4_5_6.json`

### Test 3: Interactive Analysis
```bash
jupyter notebook notebooks/data_analysis.ipynb
```
**Съдържа:** EDA, correlation analysis, visualizations  
**Output:** `notebooks/results/` (charts + HTML report)

## 📊 Expected Results

- **R² Score**: 0.9062 (90.62% variance explained)
- **RMSE**: 0.0442
- **MAE**: 0.0321
- **Dataset**: 5000 rows, 44 features (after processing)
- **Outliers**: 250 detected, imputed (not removed)

## ⚙️ Configuration

Edit в `main.py`:
```python
contamination=0.05      # Outlier detection threshold
impute_strategy="median"
feature_depth=1         # DFS depth
```

Edit в `run_experiments.py` (OPTIMIZED Grid):
```python
param_grid = {
    'n_estimators': [100, 200],    # 4 combinations total
    'max_depth': [10, None],       # (was 36, ~20 min)
    'min_samples_split': [2],      # now 4, ~1 min
    'min_samples_leaf': [1]
}
```

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: featuretools` | `pip install featuretools` |
| `ModuleNotFoundError: notebook` | `pip install jupyter notebook` |
| Missing dataset | Download from Kaggle → `data/raw/` |
| Grid Search slow | Already optimized (4 combos, ~1 min) |

## 📈 Key Features

✅ Data Leakage Prevention (target separation)  
✅ Optimized Grid Search (4 combos, 8.7x faster)  
✅ Automated Feature Engineering (DFS)  
✅ Model Comparison (4 models)  
✅ Residual Analysis (Shapiro-Wilk, Q-Q plots)  
✅ Interactive EDA (Jupyter notebook)  
✅ HTML Reports (auto-generated)  

## 📝 Notes

- **Outliers**: Imputed, not removed (5000 → 5000 rows)
- **Features**: 26 → 44 (27 generated, cleanup applied)
- **Tuning**: OPTIMIZED Grid (4 combos, not 36)
- **Notebook**: Separate EDA workspace with visualizations

## 🎓 Project Info

**Course**: Приложен изкуствен интелект  
**Dataset**: [Agentic AI Performance (Kaggle)](https://www.kaggle.com/datasets/bismasajjad/agentic-ai-performance-and-capabilities-dataset)  
**Tech Stack**: Python, scikit-learn, featuretools, pandas, matplotlib, Jupyter

---

**Quick Test**: `python main.py` → ~30s → R² ≈ 0.91 ✅