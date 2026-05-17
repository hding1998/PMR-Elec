# PMR-Elec: Experiment Code

This directory contains the code for running comparative experiments on LLM-based and traditional machine-learning methods for power-system tasks.

## Environment Setup

```bash
# Python 3.12
conda create -n pmr python=3.12
conda activate pmr

# Core dependencies
pip install numpy pandas scikit-learn torch tabnet xgboost lightgbm

# LLM API
pip install openai

# Local embedding (optional)
pip install ollama

# Visualization
pip install matplotlib seaborn
```

## Experiment Scripts

### 1. Fault Diagnosis (5-fold CV on DGA records)

```bash
python experiments/5fold_cv_real_data.py
```

Runs 5-fold stratified cross-validation with SMOTE augmentation. Methods include SVM, Random Forest, XGBoost, LightGBM, MLP, TabNet, Duval Triangle, and several LLM prompting configurations.

**Note**: LLM experiments require a DeepSeek API key set in environment variable `DEEPSEEK_API_KEY`.

### 2. Load Forecasting (multiple baselines)

```bash
python experiments/load_forecasting_baselines.py
```

Compares methods across three forecast horizons (H = 1, 3, 7): Naive, SMA, SARIMA, ARIMA, DLinear, Linear Regression, XGBoost, LLM-RAG, Historical Mean.

### 3. Structured KB RAG Experiment

```bash
python experiments/rag_60case_experiment.py
```

Validates diagnostic accuracy of a 60-case structured knowledge base versus original text-based regulation documents. Requires Ollama with `bge-m3:latest` embedding model.

### 4. Data Leakage Audit

```bash
python experiments/data_leakage_audit.py
```

Performs automated code audit of hypothesized data-leakage pathways in the load-forecasting pipeline.

## Data

- `knowledge_base/extended_kb_cases.json`: 60 structured fault-case entries for case-based retrieval experiments
- `knowledge_base/*.txt`: Original Chinese regulation documents (DL/T 722-2014, DL/T 596, etc.)
