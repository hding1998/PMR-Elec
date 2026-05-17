# PMR-Elec

[![Python](https://img.shields.io/badge/Python-3.12-green)](code/README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository contains experimental code and supporting materials for evaluating Large Language Model (LLM) retrieval-augmented generation (RAG) methods on two power-system tasks: transformer dissolved-gas-analysis (DGA) fault diagnosis and short-term load forecasting.

## Repository Structure

```
.
├── code/
│   ├── experiments/          # Experiment scripts
│   └── knowledge_base/       # Regulation documents & structured cases
├── data/
│   └── dga_transformer_data.csv   # DGA dataset sample
└── README.md                 # This file
```

## Quick Start

### Environment

```bash
conda create -n pmr python=3.12
conda activate pmr
pip install numpy pandas scikit-learn torch tabnet xgboost lightgbm matplotlib seaborn openai ollama
```

### Run Experiments

```bash
# 1. Fault diagnosis (5-fold CV on DGA records)
python code/experiments/5fold_cv_real_data.py

# 2. Load forecasting (multiple baselines)
python code/experiments/load_forecasting_baselines.py

# 3. Structured KB RAG experiment (60 cases)
python code/experiments/rag_60case_experiment.py

# 4. Data leakage audit
python code/experiments/data_leakage_audit.py
```

**Note**: LLM experiments require a DeepSeek API key (`export DEEPSEEK_API_KEY=your_key`).

## Data

- `data/dga_transformer_data.csv`: DGA dataset used for fault-diagnosis experiments.
- `code/knowledge_base/*.txt`: Chinese power-industry regulation documents (DL/T 722-2014, DL/T 596, etc.).
- `code/knowledge_base/extended_kb_cases.json`: 60 structured fault-case entries for case-based retrieval experiments.

## Contact

**Han Ding**  
State Grid Suzhou Power Supply Company  
📧 hding1998@foxmail.com
