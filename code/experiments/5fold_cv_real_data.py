#!/usr/bin/env python3
"""
真实DGA数据5-fold Stratified Cross-Validation实验

使用全部703条真实DGA数据，运行5-fold分层交叉验证，对比9种方法。
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import warnings
import logging
import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 依赖检测
try:
    from xgboost import XGBClassifier
    HAS_XGB: bool = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB: bool = True
except ImportError:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN: bool = True
except ImportError:
    HAS_IMBLEARN = False

try:
    from openai import OpenAI
    HAS_OPENAI: bool = True
except ImportError:
    HAS_OPENAI = False

# TabNet检测
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET: bool = True
except ImportError:
    HAS_TABNET = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v2_experiment_main import (
    LLMClient, PromptBuilder, FAULT_TYPES, FAULT_SHORT,
    DUVAL_RATIOS
)

# =================== 真实数据配置 ===================

REAL_FAULT_MAP = {
    'Normal': 0,
    'Partial_Discharge': 1,
    'Low_Energy_Discharge': 2,
    'High_Energy_Discharge': 3,
    'Low_Temp_Thermal': 4,
    'High_Temp_Thermal': 6,
}

FEATURE_COLS = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'Total_HC']

# =================== 数据加载与处理 ===================

def load_real_data() -> pd.DataFrame:
    """加载真实DGA数据"""
    df = pd.read_csv('data/raw/dga_transformer_data.csv')
    df['Fault_Code'] = df['Fault'].map(REAL_FAULT_MAP)
    df = df.dropna(subset=['Fault_Code'])
    df['Fault_Code'] = df['Fault_Code'].astype(int)
    logger.info(f"[真实数据] 加载 {len(df)} 条样本")
    logger.info(f"[真实数据] 类别分布: {dict(df['Fault_Code'].value_counts().sort_index())}")
    return df

def compute_duval_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算Duval特征和三比值编码"""
    df = df.copy()
    total = df["CH4"] + df["C2H6"] + df["C2H4"] + 1e-10
    df["CH4_pct"] = df["CH4"] / total * 100
    df["C2H6_pct"] = df["C2H6"] / total * 100
    df["C2H4_pct"] = df["C2H4"] / total * 100
    df["R1"] = df["C2H2"] / (df["C2H4"] + 1e-10)
    df["R2"] = df["CH4"] / (df["H2"] + 1e-10)
    df["R3"] = df["C2H2"] / (df["C2H6"] + 1e-10)
    df["R1_code"] = df["R1"].apply(lambda x: 0 if x < 0.1 else (1 if x < 3.0 else 2))
    df["R2_code"] = df["R2"].apply(lambda x: 0 if x < 0.1 else (1 if x < 1.0 else 2))
    df["R3_code"] = df["R3"].apply(lambda x: 0 if x < 0.1 else (1 if x < 3.0 else 2))
    df["duval_predict"] = df.apply(
        lambda row: DUVAL_RATIOS.get(
            (int(row["R1_code"]), int(row["R2_code"]), int(row["R3_code"])), -1
        ), axis=1
    )
    return df

def duval_triangle_rule(df: pd.DataFrame) -> np.ndarray:
    """Duval三角法故障诊断规则（IEC 60599标准）"""
    predictions: List[int] = []
    for _, row in df.iterrows():
        total = row["CH4"] + row["C2H6"] + row["C2H4"]
        if total < 1.0:
            predictions.append(0)
            continue
        ch4_pct = row["CH4"] / total * 100
        c2h6_pct = row["C2H6"] / total * 100
        c2h4_pct = row["C2H4"] / total * 100
        c2h2 = row["C2H2"]

        if ch4_pct > 98 and c2h2 < 0.5:
            predictions.append(0)
        elif c2h2 > 30 and c2h4_pct > 40:
            predictions.append(3)
        elif c2h2 > 5:
            predictions.append(2)
        elif c2h4_pct > 50:
            if c2h4_pct > 70:
                predictions.append(6)
            elif c2h4_pct > 35:
                predictions.append(5)
            else:
                predictions.append(4)
        elif ch4_pct > 50:
            if ch4_pct > 80:
                predictions.append(4)
            else:
                predictions.append(5)
        elif row["H2"] > 100 and c2h2 < 5:
            predictions.append(1)
        else:
            ratio_tuple = (int(row.get("R1_code", 0)), int(row.get("R2_code", 0)), int(row.get("R3_code", 0)))
            pred = DUVAL_RATIOS.get(ratio_tuple, 0)
            predictions.append(pred)
    return np.array(predictions)

def to_text(row: pd.Series) -> str:
    """将DGA样本转为文本描述"""
    return (
        f"变压器油中溶解气体分析结果："
        f"H2={row['H2']:.1f}μL/L, "
        f"CH4={row['CH4']:.1f}μL/L, "
        f"C2H6={row['C2H6']:.1f}μL/L, "
        f"C2H4={row['C2H4']:.1f}μL/L, "
        f"C2H2={row['C2H2']:.1f}μL/L, "
        f"总烃={row['Total_HC']:.1f}μL/L"
    )

def augment_with_smote(X_train: np.ndarray, y_train: np.ndarray, target_n: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """使用SMOTE数据增强"""
    if not HAS_IMBLEARN:
        logger.warning("[SMOTE] imblearn未安装，跳过增强")
        return X_train, y_train
    current_dist = Counter(y_train)
    n_classes = len(current_dist)
    min_target = max(max(current_dist.values()), target_n // n_classes)
    sampling_strategy = {}
    for cls, count in current_dist.items():
        if count < min_target:
            sampling_strategy[cls] = min_target
        else:
            sampling_strategy[cls] = count
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=seed,
        k_neighbors=max(1, min(5, min(current_dist.values()) - 1)),
    )
    X_aug, y_aug = smote.fit_resample(X_train, y_train)
    logger.info(f"[SMOTE] {len(y_train)} -> {len(y_aug)} 条样本")
    return X_aug, y_aug

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算fold级别的评估指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

def train_ml_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """训练所有传统ML模型并返回预测结果"""
    results: Dict[str, np.ndarray] = {}

    # SVM
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42, probability=True, class_weight='balanced')
    svm.fit(X_train, y_train)
    results['SVM'] = svm.predict(X_test)
    logger.info("[ML] SVM 完成")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    results['Random Forest'] = rf.predict(X_test)
    logger.info("[ML] Random Forest 完成")

    # XGBoost
    if HAS_XGB:
        le_xgb = LabelEncoder()
        y_train_xgb = le_xgb.fit_transform(y_train)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0, eval_metric='mlogloss',
        )
        xgb.fit(X_train, y_train_xgb)
        preds_xgb = xgb.predict(X_test)
        results['XGBoost'] = le_xgb.inverse_transform(preds_xgb)
        logger.info("[ML] XGBoost 完成")

    # LightGBM
    if HAS_LGB:
        lgb = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbose=-1,
        )
        lgb.fit(X_train, y_train)
        results['LightGBM'] = lgb.predict(X_test)
        logger.info("[ML] LightGBM 完成")

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=1000,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    mlp.fit(X_train, y_train)
    results['MLP'] = mlp.predict(X_test)
    logger.info("[ML] MLP 完成")

    return results

def train_tabnet(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Optional[np.ndarray]:
    """训练TabNet模型"""
    if not HAS_TABNET:
        return None
    try:
        import torch
        clf = TabNetClassifier(
            n_d=32, n_a=32, n_steps=3,
            gamma=1.3, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',
            verbose=0,
            seed=42,
        )
        le_tn = LabelEncoder()
        y_train_tn = le_tn.fit_transform(y_train)
        clf.fit(
            X_train, y_train_tn,
            eval_set=[(X_train, y_train_tn)],
            max_epochs=100,
            patience=10,
            batch_size=256,
        )
        preds_tn = clf.predict(X_test)
        preds = le_tn.inverse_transform(preds_tn.ravel() if hasattr(preds_tn, 'ravel') else preds_tn)
        logger.info("[ML] TabNet 完成")
        return preds
    except Exception as e:
        logger.warning(f"[TabNet] 训练失败: {e}")
        return None

def run_llm_zero_shot(test_df: pd.DataFrame, api_key: str) -> np.ndarray:
    """运行LLM Zero-shot实验"""
    if not HAS_OPENAI:
        logger.warning("[LLM] openai未安装，跳过Zero-shot")
        return np.full(len(test_df), -1)
    llm = LLMClient(model_name='deepseek-chat', api_key=api_key, temperature=0.0)
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="LLM Zero-shot"):
        text = to_text(row)
        messages = PromptBuilder.build_messages(
            model_name='deepseek-chat', text_description=text, method='zero-shot'
        )
        resp = llm.call(messages, max_tokens=10)
        numbers = re.findall(r"\b[0-6]\b", resp)
        pred = int(numbers[0]) if numbers else 0
        predictions.append(pred)
        time.sleep(0.05)  # 避免rate limit
    logger.info(f"[LLM] Zero-shot 完成, API费用: CNY {llm.total_cost:.4f}")
    return np.array(predictions)

def run_llm_few_shot(train_df: pd.DataFrame, test_df: pd.DataFrame, api_key: str, n_shots: int = 14) -> np.ndarray:
    """运行LLM Few-shot实验"""
    if not HAS_OPENAI:
        logger.warning("[LLM] openai未安装，跳过Few-shot")
        return np.full(len(test_df), -1)
    llm = LLMClient(model_name='deepseek-chat', api_key=api_key, temperature=0.0)

    # 准备分层示例
    examples = []
    n_per_class = max(1, n_shots // len(train_df['Fault_Code'].unique()))
    for fault_id in sorted(train_df['Fault_Code'].unique()):
        samples = train_df[train_df['Fault_Code'] == fault_id]
        n_select = min(n_per_class, len(samples))
        if n_select > 0:
            selected = samples.sample(n=n_select, random_state=42)
            for _, row in selected.iterrows():
                examples.append((to_text(row), int(row['Fault_Code'])))

    logger.info(f"[LLM] Few-shot示例: {len(examples)}个")

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="LLM Few-shot"):
        text = to_text(row)
        messages = PromptBuilder.build_messages(
            model_name='deepseek-chat', text_description=text,
            method='few-shot', examples=examples
        )
        resp = llm.call(messages, max_tokens=10)
        numbers = re.findall(r"\b[0-6]\b", resp)
        pred = int(numbers[0]) if numbers else 0
        predictions.append(pred)
        time.sleep(0.05)
    logger.info(f"[LLM] Few-shot 完成, API费用: CNY {llm.total_cost:.4f}")
    return np.array(predictions)

# =================== 主实验 ===================

def main():
    parser = argparse.ArgumentParser(description='5-Fold CV on Real DGA Data')
    parser.add_argument('--methods', type=str, default='all',
                        help='Methods to run: all, ML_only, or comma-separated list (Duval Triangle,SVM,Random Forest,XGBoost,LightGBM,MLP,TabNet,LLM-ZeroShot,LLM-FewShot)')
    args = parser.parse_args()

    methods_arg = args.methods.strip().lower()
    if methods_arg == 'all':
        run_methods = {'Duval Triangle', 'SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'MLP', 'TabNet', 'LLM-ZeroShot', 'LLM-FewShot'}
    elif methods_arg == 'ml_only':
        run_methods = {'Duval Triangle', 'SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'MLP', 'TabNet'}
    else:
        run_methods = set([m.strip() for m in methods_arg.split(',')])

    logger.info(f"[配置] 将运行以下方法: {run_methods}")

    # 加载API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        logger.warning("[警告] 未找到DEEPSEEK_API_KEY，LLM实验将跳过")

    # 加载数据
    df = load_real_data()
    df = compute_duval_features(df)
    X = df[FEATURE_COLS].values
    y = df['Fault_Code'].values

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 存储所有fold结果
    all_results: Dict[str, List[Dict[str, float]]] = {
        'Duval Triangle': [],
        'SVM': [],
        'Random Forest': [],
        'XGBoost': [],
        'LightGBM': [],
        'MLP': [],
        'TabNet': [],
        'LLM-ZeroShot': [],
        'LLM-FewShot': [],
    }

    fold_idx = 0
    for train_index, test_index in skf.split(X, y):
        fold_idx += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[Fold {fold_idx}/5]")
        logger.info(f"{'='*60}")

        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_df_fold = df.iloc[train_index].copy()
        test_df_fold = df.iloc[test_index].copy()

        logger.info(f"[Fold {fold_idx}] 训练集: {len(y_train)}, 测试集: {len(y_test)}")
        logger.info(f"[Fold {fold_idx}] 训练集分布: {dict(Counter(y_train))}")
        logger.info(f"[Fold {fold_idx}] 测试集分布: {dict(Counter(y_test))}")

        # ---- Duval Triangle（规则方法，不需要训练） ----
        if 'Duval Triangle' in run_methods:
            duval_pred = duval_triangle_rule(test_df_fold)
            metrics = compute_fold_metrics(y_test, duval_pred)
            all_results['Duval Triangle'].append(metrics)
            logger.info(f"[Fold {fold_idx}] Duval Triangle: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
        else:
            logger.info(f"[Fold {fold_idx}] Duval Triangle: 跳过")

        # ---- SMOTE增强 ----
        X_train_aug, y_train_aug = augment_with_smote(X_train_raw, y_train, target_n=2000, seed=42)

        # ---- 标准化 ----
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_aug)
        X_test_s = scaler.transform(X_test_raw)

        # ---- ML模型 ----
        if any(m in run_methods for m in ['SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'MLP']):
            ml_preds = train_ml_models(X_train_s, y_train_aug, X_test_s)
            for method_name, y_pred in ml_preds.items():
                if method_name in run_methods:
                    metrics = compute_fold_metrics(y_test, y_pred)
                    all_results[method_name].append(metrics)
                    logger.info(f"[Fold {fold_idx}] {method_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                else:
                    logger.info(f"[Fold {fold_idx}] {method_name}: 跳过")
        else:
            logger.info(f"[Fold {fold_idx}] ML models: 全部跳过")

        # ---- TabNet ----
        if 'TabNet' in run_methods:
            if HAS_TABNET:
                tabnet_pred = train_tabnet(X_train_s, y_train_aug, X_test_s)
                if tabnet_pred is not None:
                    metrics = compute_fold_metrics(y_test, tabnet_pred)
                    all_results['TabNet'].append(metrics)
                    logger.info(f"[Fold {fold_idx}] TabNet: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                else:
                    logger.warning(f"[Fold {fold_idx}] TabNet训练失败")
            else:
                logger.info(f"[Fold {fold_idx}] TabNet: 跳过（未安装）")
        else:
            logger.info(f"[Fold {fold_idx}] TabNet: 跳过")

        # ---- LLM ZeroShot ----
        if 'LLM-ZeroShot' in run_methods:
            if api_key:
                try:
                    llm_zs_pred = run_llm_zero_shot(test_df_fold, api_key)
                    metrics = compute_fold_metrics(y_test, llm_zs_pred)
                    all_results['LLM-ZeroShot'].append(metrics)
                    logger.info(f"[Fold {fold_idx}] LLM-ZeroShot: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                except Exception as e:
                    logger.error(f"[Fold {fold_idx}] LLM-ZeroShot失败: {e}")
            else:
                logger.info(f"[Fold {fold_idx}] LLM-ZeroShot: 跳过（无API key）")
        else:
            logger.info(f"[Fold {fold_idx}] LLM-ZeroShot: 跳过")

        # ---- LLM FewShot ----
        if 'LLM-FewShot' in run_methods:
            if api_key:
                try:
                    llm_fs_pred = run_llm_few_shot(train_df_fold, test_df_fold, api_key, n_shots=14)
                    metrics = compute_fold_metrics(y_test, llm_fs_pred)
                    all_results['LLM-FewShot'].append(metrics)
                    logger.info(f"[Fold {fold_idx}] LLM-FewShot: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                except Exception as e:
                    logger.error(f"[Fold {fold_idx}] LLM-FewShot失败: {e}")
            else:
                logger.info(f"[Fold {fold_idx}] LLM-FewShot: 跳过（无API key）")
        else:
            logger.info(f"[Fold {fold_idx}] LLM-FewShot: 跳过")

    # =================== 汇总结果 ===================
    logger.info(f"\n{'='*60}")
    logger.info("5-Fold Cross-Validation 结果汇总")
    logger.info(f"{'='*60}")

    summary_rows = []
    for method_name, fold_metrics in all_results.items():
        if not fold_metrics:
            continue
        accs = [m['accuracy'] for m in fold_metrics]
        f1s = [m['f1_macro'] for m in fold_metrics]
        kappas = [m['kappa'] for m in fold_metrics]
        mccs = [m['mcc'] for m in fold_metrics]

        row = {
            'Method': method_name,
            'Accuracy_mean': np.mean(accs),
            'Accuracy_std': np.std(accs),
            'F1_Macro_mean': np.mean(f1s),
            'F1_Macro_std': np.std(f1s),
            'Kappa_mean': np.mean(kappas),
            'Kappa_std': np.std(kappas),
            'MCC_mean': np.mean(mccs),
            'MCC_std': np.std(mccs),
        }
        summary_rows.append(row)
        logger.info(f"{method_name:20s} | Acc: {row['Accuracy_mean']:.4f}±{row['Accuracy_std']:.4f} | F1: {row['F1_Macro_mean']:.4f}±{row['F1_Macro_std']:.4f} | Kappa: {row['Kappa_mean']:.4f}±{row['Kappa_std']:.4f} | MCC: {row['MCC_mean']:.4f}±{row['MCC_std']:.4f}")

    summary_df = pd.DataFrame(summary_rows)

    # 保存CSV
    output_dir = Path('paper_materials/experiment_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / '5fold_cv_real_data_results.csv'
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"\n[保存] CSV结果已保存至 {csv_path}")

    # 生成Markdown表格
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
    if not HAS_TABNET:
        md_lines.append("- TabNet: not available in current environment")

    md_path = output_dir / '5fold_cv_results_table.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    logger.info(f"[保存] Markdown表格已保存至 {md_path}")

    # 保存详细JSON
    class_dist = {int(k): int(v) for k, v in df['Fault_Code'].value_counts().sort_index().items()}
    detail = {
        'dataset': 'real_dga_703',
        'n_folds': 5,
        'n_samples': int(len(df)),
        'class_distribution': class_dist,
        'summary': summary_df.to_dict('records'),
        'fold_details': {
            method: [
                {k: round(float(v), 6) for k, v in m.items()}
                for m in metrics_list
            ]
            for method, metrics_list in all_results.items() if metrics_list
        }
    }
    json_path = output_dir / '5fold_cv_real_data_detail.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    logger.info(f"[保存] 详细JSON已保存至 {json_path}")

    logger.info("\n[完成] 实验全部结束！")
    return summary_df


if __name__ == '__main__':
    main()
