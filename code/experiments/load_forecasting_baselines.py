#!/usr/bin/env python3
"""
负荷预测基线实验 — 基于真实 household_power_consumption 数据

补充缺失基线：
1. Naive Persistence
2. Historical Mean
3. SARIMA
4. DLinear (Decomposition + Linear, numpy实现)
5. Simple Moving Average

同时复现现有基线：ARIMA, XGBoost, Linear Regression, LLM-RAG
"""

import os
import sys
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =================== 数据加载与预处理 ===================

def load_and_preprocess_data() -> pd.DataFrame:
    """加载 household_power_consumption.txt 并预处理为日级数据"""
    data_path = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
    
    print(f"[Data] Loading {data_path} ...")
    df = pd.read_csv(data_path, sep=';', na_values=['?', 'NA', ''])
    
    # 合并日期时间
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime'])
    
    # 转换数值列
    num_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 按日聚合: 使用 Global_active_power 的日均值作为目标
    df['Date'] = df['DateTime'].dt.date
    daily = df.groupby('Date').agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
    }).reset_index()
    daily.columns = ['Date', 'Global_active_power', 'Global_reactive_power',
                     'Voltage', 'Global_intensity', 'Sub_metering_1',
                     'Sub_metering_2', 'Sub_metering_3']
    
    # 去除缺失日
    daily = daily.dropna(subset=['Global_active_power']).reset_index(drop=True)
    daily['Date'] = pd.to_datetime(daily['Date'])
    
    # 时间特征
    daily['day_of_week'] = daily['Date'].dt.dayofweek  # 0=Mon
    daily['is_weekend'] = (daily['day_of_week'] >= 5).astype(int)
    daily['month'] = daily['Date'].dt.month
    daily['day_of_year'] = daily['Date'].dt.dayofyear
    
    print(f"[Data] Daily records: {len(daily)}, Range: {daily['Date'].min().date()} ~ {daily['Date'].max().date()}")
    print(f"[Data] Global_active_power stats: mean={daily['Global_active_power'].mean():.3f}, std={daily['Global_active_power'].std():.3f}")
    return daily


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建ML特征"""
    df = df.copy()
    target = 'Global_active_power'
    
    for lag in [1, 2, 3, 7, 14]:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)
    for win in [3, 7, 14]:
        df[f'{target}_ma{win}'] = df[target].shift(1).rolling(win, min_periods=1).mean()
    # 注意: diff特征在测试时会导致数据泄露（使用当前行的真实值），故移除
    for col in ['Global_reactive_power', 'Voltage', 'Global_intensity']:
        df[f'{col}_lag1'] = df[col].shift(1)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 移除同时刻的泄露特征（这些变量与目标有极强的物理相关性）
    df = df.drop(columns=['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], errors='ignore')
    df = df.dropna().reset_index(drop=True)
    return df


# =================== 评估指标 ===================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# =================== 基线方法 ===================

def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Naive Persistence: 用前H天的值作为预测值"""
    last = train['Global_active_power'].iloc[-horizon:].values
    if len(last) < horizon:
        last = np.pad(last, (horizon - len(last), 0), mode='edge')
    return last


def historical_mean_forecast(train: pd.DataFrame, test_slice: pd.DataFrame, horizon: int) -> np.ndarray:
    """Historical Mean: 使用训练集中同期（同一星期几）的历史平均值"""
    preds = []
    for i in range(horizon):
        dow = test_slice.iloc[i]['day_of_week']
        hist_mean = train[train['day_of_week'] == dow]['Global_active_power'].mean()
        preds.append(hist_mean)
    return np.array(preds)


def sma_forecast(train: pd.DataFrame, horizon: int, window: int = 7) -> np.ndarray:
    """Simple Moving Average"""
    pred = train['Global_active_power'].tail(window).mean()
    return np.full(horizon, pred)


def arima_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    try:
        series = train['Global_active_power'].values
        model = ARIMA(series, order=(2, 1, 1))
        fitted = model.fit()
        return fitted.forecast(steps=horizon)
    except Exception as e:
        print(f"  [WARN] ARIMA failed: {e}")
        last = train['Global_active_power'].iloc[-1]
        return np.full(horizon, last)


def sarima_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    try:
        series = train['Global_active_power'].values
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        return fitted.forecast(steps=horizon)
    except Exception as e:
        print(f"  [WARN] SARIMA failed: {e}, fallback ARIMA")
        return arima_forecast(train, horizon)


def _ml_multi_step(train: pd.DataFrame, test_slice: pd.DataFrame, horizon: int, model) -> np.ndarray:
    """通用ML多步预测"""
    train_feat = build_ml_features(train)
    feature_cols = [c for c in train_feat.columns if c not in ['Date', 'Global_active_power']]
    X_train = train_feat[feature_cols].values
    y_train = train_feat['Global_active_power'].values
    model.fit(X_train, y_train)
    
    preds = []
    current = train.copy()
    for i in range(horizon):
        next_row = test_slice.iloc[[i]].copy()
        current = pd.concat([current, next_row], ignore_index=True)
        current_feat = build_ml_features(current)
        X_pred = current_feat[feature_cols].iloc[[-1]].values
        pred = float(model.predict(X_pred)[0])
        preds.append(pred)
        current.loc[current.index[-1], 'Global_active_power'] = pred
    return np.array(preds)


def xgboost_forecast(train: pd.DataFrame, test_slice: pd.DataFrame, horizon: int) -> np.ndarray:
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=2, verbosity=0)
    return _ml_multi_step(train, test_slice, horizon, model)


def linear_forecast(train: pd.DataFrame, test_slice: pd.DataFrame, horizon: int) -> np.ndarray:
    model = LinearRegression()
    return _ml_multi_step(train, test_slice, horizon, model)


def dlinear_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """DLinear: Decomposition + Linear (numpy实现)"""
    series = train['Global_active_power'].values
    n = len(series)
    
    # 分解
    decomp_window = 7
    trend = pd.Series(series).rolling(decomp_window, min_periods=1, center=True).mean().values
    seasonal = series - trend
    
    look_back = min(30, n - 1)
    if look_back < 2:
        return np.full(horizon, series[-1])
    
    # 趋势线性层
    X_trend, y_trend = [], []
    for i in range(look_back, n - horizon + 1):
        X_trend.append(trend[i - look_back:i])
        y_trend.append(trend[i:i + horizon])
    X_trend = np.array(X_trend)
    y_trend = np.array(y_trend)
    
    if len(X_trend) > 0:
        model_trend = LinearRegression()
        model_trend.fit(X_trend, y_trend)
        pred_trend = model_trend.predict(trend[-look_back:].reshape(1, -1))[0]
    else:
        pred_trend = np.full(horizon, trend[-1])
    
    # 季节线性层
    X_seas, y_seas = [], []
    for i in range(look_back, n - horizon + 1):
        X_seas.append(seasonal[i - look_back:i])
        y_seas.append(seasonal[i:i + horizon])
    X_seas = np.array(X_seas)
    y_seas = np.array(y_seas)
    
    if len(X_seas) > 0:
        model_seas = LinearRegression()
        model_seas.fit(X_seas, y_seas)
        pred_seas = model_seas.predict(seasonal[-look_back:].reshape(1, -1))[0]
    else:
        pred_seas = np.full(horizon, seasonal[-1])
    
    return pred_trend + pred_seas


# =================== LLM 预测 ===================

RAG_KNOWLEDGE = """
【电力负荷预测方法】
1. 相似日法：选取与预测日气象特征相似的历史日作为参考。
2. 时间序列法：ARIMA适合短期预测，SARIMA考虑季节因素。
3. 机器学习：XGBoost、LSTM等需特征工程。
4. 峰谷特征：工作日负荷曲线双峰（上午、下午），周末单峰且峰值较低。

【电力负荷规律】
- 家庭用电：工作日白天用电低、晚上高；周末白天用电较高。
- 季节因素：夏季空调、冬季采暖导致负荷波动。
- 滞后效应：连续高温后第2天负荷常高于第1天。
"""


def get_llm_client():
    try:
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None, None
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), "deepseek-chat"
    except Exception:
        return None, None


def llm_rag_forecast(history: pd.DataFrame, horizon: int, client, model_name: str) -> np.ndarray:
    if client is None:
        return None
    system_prompt = "你是一位资深电力负荷预测专家。根据历史负荷数据，预测未来几天的日平均负荷（单位：kW）。只输出预测数值，格式为逗号分隔，如：4.2, 5.1, ..."
    history_text = "\n".join(
        f"{row['Date'].strftime('%Y-%m-%d')} 星期{int(row['day_of_week'])+1}: 负荷={row['Global_active_power']:.3f}kW"
        for _, row in history.iterrows()
    )
    user_prompt = (f"参考电力负荷预测知识：\n{RAG_KNOWLEDGE}\n\n"
                   f"历史{len(history)}天数据：\n{history_text}\n\n"
                   f"请预测接下来{horizon}天的日平均负荷（单位：kW），只输出数值，逗号分隔：")
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, max_tokens=200,
        )
        result = resp.choices[0].message.content.strip()
        numbers = re.findall(r'\d+\.?\d*', result)
        preds = [float(n) for n in numbers[:horizon]]
        while len(preds) < horizon:
            preds.append(preds[-1] if preds else history['Global_active_power'].mean())
        return np.array(preds[:horizon])
    except Exception as e:
        print(f"  [WARN] LLM API error: {e}")
        return None


# =================== 实验主流程 ===================

def run_experiment():
    print("=" * 70)
    print("负荷预测基线实验 — 基于真实 household_power_consumption 数据")
    print("=" * 70)
    
    df = load_and_preprocess_data()
    
    # 划分训练/测试集: 80/20
    n_total = len(df)
    n_train = int(n_total * 0.8)
    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    test_df = df.iloc[n_train:].copy().reset_index(drop=True)
    print(f"[Split] Train: {len(train_df)} days, Test: {len(test_df)} days")
    
    # LLM client
    llm_client, llm_model = get_llm_client()
    if llm_client:
        print("[LLM] DeepSeek API connected")
    else:
        print("[LLM] DeepSeek API not available, will skip LLM-RAG")
    
    horizons = [1, 3, 7]
    methods_results = []
    
    # 为了控制总时间，每个horizon评估约30个窗口
    max_windows_per_horizon = 30
    
    for horizon in horizons:
        print(f"\n{'-' * 70}")
        print(f"Horizon H = {horizon}")
        print(f"{'-' * 70}")
        
        max_start = len(test_df) - horizon + 1
        # 均匀采样测试起点
        if max_start <= max_windows_per_horizon:
            test_starts = list(range(max_start))
        else:
            stride = max(1, max_start // max_windows_per_horizon)
            test_starts = list(range(0, max_start, stride))[:max_windows_per_horizon]
        n_eval = len(test_starts)
        print(f"[Eval] Evaluating {n_eval} windows (sampled from {max_start} possible)")
        
        all_preds = {m: [] for m in ['Naive', 'Historical Mean', 'SMA', 'ARIMA', 'SARIMA',
                                      'DLinear', 'Linear Regression', 'XGBoost', 'LLM-RAG']}
        all_truths = []
        
        for idx, start in enumerate(test_starts):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{n_eval}")
            
            current_train = pd.concat([train_df, test_df.iloc[:start]], ignore_index=True)
            test_slice = test_df.iloc[start:start + horizon].copy().reset_index(drop=True)
            truth = test_slice['Global_active_power'].values
            all_truths.append(truth)
            
            # Naive
            all_preds['Naive'].append(naive_forecast(current_train, horizon))
            
            # Historical Mean
            all_preds['Historical Mean'].append(historical_mean_forecast(current_train, test_slice, horizon))
            
            # SMA
            all_preds['SMA'].append(sma_forecast(current_train, horizon, window=7))
            
            # ARIMA
            all_preds['ARIMA'].append(arima_forecast(current_train, horizon))
            
            # SARIMA
            all_preds['SARIMA'].append(sarima_forecast(current_train, horizon))
            
            # DLinear
            all_preds['DLinear'].append(dlinear_forecast(current_train, horizon))
            
            # Linear Regression
            all_preds['Linear Regression'].append(linear_forecast(current_train, test_slice, horizon))
            
            # XGBoost
            all_preds['XGBoost'].append(xgboost_forecast(current_train, test_slice, horizon))
            
            # LLM-RAG (每3个窗口跑一次)
            if llm_client and idx % 3 == 0:
                history_llm = current_train.tail(30)
                preds_llm = llm_rag_forecast(history_llm, horizon, llm_client, llm_model)
                if preds_llm is not None:
                    all_preds['LLM-RAG'].append(preds_llm)
                else:
                    all_preds['LLM-RAG'].append(np.full(horizon, np.nan))
            else:
                all_preds['LLM-RAG'].append(np.full(horizon, np.nan))
        
        # 汇总指标
        truths_arr = np.array(all_truths).flatten()
        
        for method_name in all_preds:
            preds_arr = np.array(all_preds[method_name])
            if method_name == 'LLM-RAG':
                valid_mask = ~np.isnan(preds_arr.flatten())
                if valid_mask.sum() == 0:
                    print(f"  {method_name}: SKIPPED (no valid predictions)")
                    continue
                y_t = truths_arr[valid_mask]
                y_p = preds_arr.flatten()[valid_mask]
            else:
                y_t = truths_arr
                y_p = preds_arr.flatten()
            
            metrics = compute_metrics(y_t, y_p)
            result = {
                "Horizon": horizon,
                "Method": method_name,
                **{k: round(v, 4) for k, v in metrics.items()}
            }
            methods_results.append(result)
            print(f"  {method_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.4f}%, R2={metrics['R2']:.4f}")
    
    # 保存结果
    result_df = pd.DataFrame(methods_results)
    
    out_dir = PROJECT_ROOT / "paper_materials" / "experiment_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "load_forecasting_baselines.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Save] CSV saved to {csv_path}")
    
    # Markdown表格
    md_path = out_dir / "load_forecasting_baselines_table.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 负荷预测基线实验结果\n\n")
        f.write("基于 household_power_consumption.txt 真实数据（日级平均负荷，单位：kW）\n\n")
        
        for horizon in horizons:
            f.write(f"## Horizon H = {horizon}\n\n")
            sub = result_df[result_df['Horizon'] == horizon].copy()
            sub = sub.sort_values('MAPE')
            f.write("| Method | MAE | RMSE | MAPE(%) | R² |\n")
            f.write("|--------|-----|------|---------|-----|\n")
            for _, row in sub.iterrows():
                f.write(f"| {row['Method']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['MAPE']:.4f} | {row['R2']:.4f} |\n")
            f.write("\n")
        
        f.write("## 关键发现\n\n")
        
        def get_val(horizon, method, metric):
            vals = result_df[(result_df['Horizon'] == horizon) & (result_df['Method'] == method)][metric].values
            return vals[0] if len(vals) > 0 else None
        
        naive_mape = get_val(3, 'Naive', 'MAPE')
        llm_mape = get_val(3, 'LLM-RAG', 'MAPE')
        sarima_mape = get_val(3, 'SARIMA', 'MAPE')
        arima_mape = get_val(3, 'ARIMA', 'MAPE')
        hm_mape = get_val(3, 'Historical Mean', 'MAPE')
        
        if naive_mape is not None:
            f.write(f"1. **Naive Persistence (H=3)**: MAPE = {naive_mape:.4f}%\n")
            f.write(f"   - 若该值接近 LLM-RAG 的 MAPE，说明 LLM-RAG 可能只是在做一个简单的 persistence forecast。\n\n")
        if sarima_mape is not None and arima_mape is not None:
            f.write(f"2. **SARIMA vs ARIMA (H=3)**: SARIMA MAPE = {sarima_mape:.4f}%, ARIMA MAPE = {arima_mape:.4f}%\n")
            if sarima_mape < arima_mape:
                f.write(f"   - ✅ SARIMA 优于 ARIMA，季节性建模有效。\n\n")
            else:
                f.write(f"   - ⚠️ SARIMA 未显著优于 ARIMA。\n\n")
        if hm_mape is not None:
            f.write(f"3. **Historical Mean (H=3)**: MAPE = {hm_mape:.4f}%\n")
            f.write(f"   - 反映周期性规律的捕捉能力。\n\n")
    
    print(f"[Save] Markdown saved to {md_path}")
    
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
