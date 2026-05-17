#!/usr/bin/env python3
"""
真实DGA数据5-fold CV — LLM部分（支持断点续传）

用法:
  python 5fold_cv_llm.py --fold 1 --method zero-shot
  python 5fold_cv_llm.py --fold 1 --method few-shot
  python 5fold_cv_llm.py --merge
"""

import os
import sys
import re
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from v2_experiment_main import LLMClient, PromptBuilder

REAL_FAULT_MAP = {
    'Normal': 0,
    'Partial_Discharge': 1,
    'Low_Energy_Discharge': 2,
    'High_Energy_Discharge': 3,
    'Low_Temp_Thermal': 4,
    'High_Temp_Thermal': 6,
}
FEATURE_COLS = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'Total_HC']

def load_data():
    df = pd.read_csv('data/raw/dga_transformer_data.csv')
    df['Fault_Code'] = df['Fault'].map(REAL_FAULT_MAP)
    df = df.dropna(subset=['Fault_Code']).astype({'Fault_Code': int})
    return df

def to_text(row):
    return (
        f"变压器油中溶解气体分析结果："
        f"H2={row['H2']:.1f}μL/L, "
        f"CH4={row['CH4']:.1f}μL/L, "
        f"C2H6={row['C2H6']:.1f}μL/L, "
        f"C2H4={row['C2H4']:.1f}μL/L, "
        f"C2H2={row['C2H2']:.1f}μL/L, "
        f"总烃={row['Total_HC']:.1f}μL/L"
    )

def get_cache_path(fold, method):
    cache_dir = Path('paper_materials/experiment_data/llm_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"fold{fold}_{method}.json"

def load_cache(fold, method):
    path = get_cache_path(fold, method)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_cache(fold, method, data):
    path = get_cache_path(fold, method)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_zero_shot(test_df, api_key, fold):
    cache = load_cache(fold, 'zero-shot')
    predictions = cache['predictions'] if cache else [-1] * len(test_df)
    total_cost = cache.get('total_cost', 0.0) if cache else 0.0
    
    if all(p != -1 for p in predictions):
        print(f"[Fold {fold}] Zero-shot 缓存命中，跳过")
        return np.array(predictions), total_cost
    
    llm = LLMClient(model_name='deepseek-chat', api_key=api_key, temperature=0.0)
    llm.total_cost = total_cost
    
    rows = list(test_df.iterrows())
    for i, (_, row) in enumerate(rows):
        if predictions[i] != -1:
            continue
        text = to_text(row)
        messages = PromptBuilder.build_messages(
            model_name='deepseek-chat', text_description=text, method='zero-shot'
        )
        resp = llm.call(messages, max_tokens=10)
        numbers = re.findall(r"\b[0-6]\b", resp)
        pred = int(numbers[0]) if numbers else 0
        predictions[i] = pred
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            save_cache(fold, 'zero-shot', {'predictions': predictions, 'total_cost': llm.total_cost})
        time.sleep(0.05)
    
    save_cache(fold, 'zero-shot', {'predictions': predictions, 'total_cost': llm.total_cost})
    print(f"[Fold {fold}] Zero-shot 完成，费用: CNY {llm.total_cost:.4f}")
    return np.array(predictions), llm.total_cost

def run_few_shot(train_df, test_df, api_key, fold, n_shots=14):
    cache = load_cache(fold, 'few-shot')
    predictions = cache['predictions'] if cache else [-1] * len(test_df)
    total_cost = cache.get('total_cost', 0.0) if cache else 0.0
    
    if all(p != -1 for p in predictions):
        print(f"[Fold {fold}] Few-shot 缓存命中，跳过")
        return np.array(predictions), total_cost
    
    llm = LLMClient(model_name='deepseek-chat', api_key=api_key, temperature=0.0)
    llm.total_cost = total_cost
    
    examples = []
    n_per_class = max(1, n_shots // len(train_df['Fault_Code'].unique()))
    for fault_id in sorted(train_df['Fault_Code'].unique()):
        samples = train_df[train_df['Fault_Code'] == fault_id]
        n_select = min(n_per_class, len(samples))
        if n_select > 0:
            selected = samples.sample(n=n_select, random_state=42)
            for _, row in selected.iterrows():
                examples.append((to_text(row), int(row['Fault_Code'])))
    
    print(f"[Fold {fold}] Few-shot examples: {len(examples)}")
    
    rows = list(test_df.iterrows())
    for i, (_, row) in enumerate(rows):
        if predictions[i] != -1:
            continue
        text = to_text(row)
        messages = PromptBuilder.build_messages(
            model_name='deepseek-chat', text_description=text, method='few-shot', examples=examples
        )
        resp = llm.call(messages, max_tokens=10)
        numbers = re.findall(r"\b[0-6]\b", resp)
        pred = int(numbers[0]) if numbers else 0
        predictions[i] = pred
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            save_cache(fold, 'few-shot', {'predictions': predictions, 'total_cost': llm.total_cost})
        time.sleep(0.05)
    
    save_cache(fold, 'few-shot', {'predictions': predictions, 'total_cost': llm.total_cost})
    print(f"[Fold {fold}] Few-shot 完成，费用: CNY {llm.total_cost:.4f}")
    return np.array(predictions), llm.total_cost

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

def run_fold(fold_idx, method, api_key):
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df['Fault_Code'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    
    if fold_idx < 1 or fold_idx > 5:
        raise ValueError("fold must be 1-5")
    
    train_index, test_index = splits[fold_idx - 1]
    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()
    y_test = y[test_index]
    
    print(f"[Fold {fold_idx}] 训练集: {len(train_df)}, 测试集: {len(test_df)}")
    
    if method == 'zero-shot':
        preds, cost = run_zero_shot(test_df, api_key, fold_idx)
    elif method == 'few-shot':
        preds, cost = run_few_shot(train_df, test_df, api_key, fold_idx)
    else:
        raise ValueError("method must be zero-shot or few-shot")
    
    metrics = compute_metrics(y_test, preds)
    print(f"[Fold {fold_idx}] {method}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}, Kappa={metrics['kappa']:.4f}, MCC={metrics['mcc']:.4f}, Cost={cost:.4f}")
    return metrics

def merge_results():
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df['Fault_Code'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    
    llm_results = {'LLM-ZeroShot': [], 'LLM-FewShot': []}
    for fold_idx in range(1, 6):
        _, test_index = splits[fold_idx - 1]
        y_test = y[test_index]
        
        for method, key in [('zero-shot', 'LLM-ZeroShot'), ('few-shot', 'LLM-FewShot')]:
            cache = load_cache(fold_idx, method)
            if cache and all(p != -1 for p in cache['predictions']):
                preds = np.array(cache['predictions'])
                metrics = compute_metrics(y_test, preds)
                llm_results[key].append(metrics)
                print(f"[Merge] Fold {fold_idx} {key}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
            else:
                print(f"[Merge] Fold {fold_idx} {key}: 数据不完整，跳过")
    
    # 加载ML结果
    ml_csv = Path('paper_materials/experiment_data/5fold_cv_real_data_results.csv')
    if ml_csv.exists():
        ml_df = pd.read_csv(ml_csv)
    else:
        ml_df = pd.DataFrame()
    
    summary_rows = ml_df.to_dict('records') if not ml_df.empty else []
    
    for method_name, fold_metrics in llm_results.items():
        if not fold_metrics:
            continue
        summary_rows.append({
            'Method': method_name,
            'Accuracy_mean': np.mean([m['accuracy'] for m in fold_metrics]),
            'Accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
            'F1_Macro_mean': np.mean([m['f1_macro'] for m in fold_metrics]),
            'F1_Macro_std': np.std([m['f1_macro'] for m in fold_metrics]),
            'Kappa_mean': np.mean([m['kappa'] for m in fold_metrics]),
            'Kappa_std': np.std([m['kappa'] for m in fold_metrics]),
            'MCC_mean': np.mean([m['mcc'] for m in fold_metrics]),
            'MCC_std': np.std([m['mcc'] for m in fold_metrics]),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    outdir = Path('paper_materials/experiment_data')
    csv_path = outdir / '5fold_cv_real_data_results.csv'
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"[保存] CSV: {csv_path}")
    
    # Markdown
    md_lines = []
    md_lines.append("# 5-Fold Stratified Cross-Validation Results on Real DGA Data (n=703)\n")
    md_lines.append("| Method | Accuracy | F1-Macro | Kappa | MCC |")
    md_lines.append("|--------|----------|----------|-------|-----|")
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['Method']} | "
            f"{row['Accuracy_mean']:.4f}±{row['Accuracy_std']:.4f} | "
            f"{row['F1_Macro_mean']:.4f}±{row['F1_Macro_std']:.4f} | "
            f"{row['Kappa_mean']:.4f}±{row['Kappa_std']:.4f} | "
            f"{row['MCC_mean']:.4f}±{row['MCC_std']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- Dataset: 703 real DGA samples from transformer fault diagnosis records")
    md_lines.append("- Cross-validation: 5-fold stratified, shuffle=True, random_state=42")
    md_lines.append("- ML models trained with SMOTE augmentation (~2000 samples per fold training set)")
    md_lines.append("- LLM: DeepSeek-V3 (deepseek-chat), temperature=0")
    md_lines.append("- Few-shot: 14 stratified examples (2 per class) selected from each fold's training set")
    if len(llm_results['LLM-ZeroShot']) < 5:
        md_lines.append("- LLM-ZeroShot: partial results (some folds pending)")
    if len(llm_results['LLM-FewShot']) < 5:
        md_lines.append("- LLM-FewShot: partial results (some folds pending)")
    
    md_path = outdir / '5fold_cv_results_table.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"[保存] Markdown: {md_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='Fold index (1-5)')
    parser.add_argument('--method', type=str, default='', help='Method: zero-shot or few-shot')
    parser.add_argument('--merge', action='store_true', help='Merge all results')
    args = parser.parse_args()
    
    if args.merge:
        merge_results()
        return
    
    if not args.fold or not args.method:
        print("Usage: python 5fold_cv_llm.py --fold 1 --method zero-shot")
        print("       python 5fold_cv_llm.py --merge")
        return
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("[错误] 未找到DEEPSEEK_API_KEY")
        return
    
    run_fold(args.fold, args.method, api_key)

if __name__ == '__main__':
    main()
