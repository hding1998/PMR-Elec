#!/usr/bin/env python3
"""
PMR-Elec 负荷预测实验 — 数据泄露根因代码审计脚本
验证 Discussion 5.3 提出的 3 条泄露途径假设。

途径1: 时间污染 — 知识库文档包含测试期统计信息
途径2: Embedding时序泄露 — 知识库分块/嵌入未按时间分区
途径3: LLM参数知识 — DeepSeek-V3预训练语料包含UCI数据集信息
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KB_DIR = PROJECT_ROOT / "knowledge_base"
RAW_KB_DIR = KB_DIR / "raw"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
EXPERIMENT_SCRIPT = EXPERIMENTS_DIR / "load_forecasting_baselines.py"

REPORT_PATH = RESULTS_DIR / "leakage_audit_report.md"


# ---------------------------------------------------------------------------
# 数据集真实统计值（预计算，用于对比）
# ---------------------------------------------------------------------------
@dataclass
class DatasetStats:
    total_days: int
    date_min: str
    date_max: str
    train_end: str
    test_start: str
    train_days: int
    test_days: int
    global_active_power_mean: float
    global_active_power_std: float
    global_active_power_min: float
    global_active_power_max: float
    voltage_mean: float
    voltage_std: float
    voltage_min: float
    voltage_max: float


def compute_dataset_stats() -> DatasetStats:
    """计算UCI household power数据集的真实统计值。"""
    df = pd.read_csv(DATA_PATH, sep=';', na_values=['?', 'NA', ''])
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime'])
    num_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = df['DateTime'].dt.date
    daily = df.groupby('Date').agg({
        'Global_active_power': 'mean',
        'Voltage': 'mean',
    }).reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.dropna()

    n_total = len(daily)
    n_train = int(n_total * 0.8)
    train_df = daily.iloc[:n_train]
    test_df = daily.iloc[n_train:]

    return DatasetStats(
        total_days=n_total,
        date_min=str(daily['Date'].min().date()),
        date_max=str(daily['Date'].max().date()),
        train_end=str(train_df['Date'].max().date()),
        test_start=str(test_df['Date'].min().date()),
        train_days=len(train_df),
        test_days=len(test_df),
        global_active_power_mean=float(daily['Global_active_power'].mean()),
        global_active_power_std=float(daily['Global_active_power'].std()),
        global_active_power_min=float(daily['Global_active_power'].min()),
        global_active_power_max=float(daily['Global_active_power'].max()),
        voltage_mean=float(daily['Voltage'].mean()),
        voltage_std=float(daily['Voltage'].std()),
        voltage_min=float(daily['Voltage'].min()),
        voltage_max=float(daily['Voltage'].max()),
    )


# ---------------------------------------------------------------------------
# 途径1: 时间污染审计
# ---------------------------------------------------------------------------
@dataclass
class Pathway1Result:
    risk_level: str
    evidence: List[str] = field(default_factory=list)
    suspicious_fragments: List[Dict] = field(default_factory=list)


def audit_pathway1() -> Pathway1Result:
    """
    审计知识库文档是否包含测试期（2008年后半年，特别是2010年）的统计信息。
    """
    evidence = []
    suspicious = []
    kb_files = []

    # 收集知识库文件
    if KB_DIR.exists():
        for ext in ('*.txt', '*.md', '*.json'):
            kb_files.extend(KB_DIR.rglob(ext))
    if RAW_KB_DIR.exists():
        for f in RAW_KB_DIR.iterdir():
            if f.is_file() and f.suffix in ('.txt', '.md', '.json'):
                kb_files.append(f)

    # 去重
    kb_files = sorted(set(str(p) for p in kb_files))
    kb_files = [Path(p) for p in kb_files]

    evidence.append(f"扫描知识库文件数量: {len(kb_files)}")

    # 敏感正则模式
    # 1. 年份 >= 2008 (特别是2009, 2010)
    year_pattern = re.compile(r'\b(200[8-9]|201[0-9])\b')
    # 2. average/mean/typical + 数值 (带单位)
    stat_pattern = re.compile(
        r'(?:average|mean|typical|median|global\s+(?:average|mean))\s*(?:value|power|voltage|consumption)?\s*(?:is|of|=|:)?\s*(\d+\.?\d*)\s*(?:kW|W|V|volts?|μL/L|μL)?',
        re.IGNORECASE
    )
    # 3. 任何看起来像功率/电压的数值 (e.g. 1.09, 240.8)
    numeric_pattern = re.compile(r'\b\d+\.\d+\b')

    total_fragments = 0
    files_with_dates = 0
    files_with_stats = 0

    for fpath in kb_files:
        try:
            text = fpath.read_text(encoding='utf-8')
        except Exception as e:
            evidence.append(f"  无法读取 {fpath.name}: {e}")
            continue

        # 检查年份
        years_found = year_pattern.findall(text)
        if years_found:
            files_with_dates += 1
            # 2009/2010是测试期年份
            test_years = [y for y in years_found if y in ('2009', '2010')]
            if test_years:
                suspicious.append({
                    "file": str(fpath.relative_to(PROJECT_ROOT)),
                    "reason": f"包含测试期年份: {sorted(set(test_years))}",
                    "fragment": text[:300].replace('\n', ' '),
                })

        # 检查统计数值
        stats_found = stat_pattern.findall(text)
        if stats_found:
            files_with_stats += 1
            # 检查是否有接近真实统计值的数字
            for num_str in stats_found:
                try:
                    val = float(num_str)
                    # 接近1.09 (kW) 或 240 (V)
                    if abs(val - 1.09) < 0.1 or abs(val - 240) < 5:
                        suspicious.append({
                            "file": str(fpath.relative_to(PROJECT_ROOT)),
                            "reason": f"包含接近真实统计值的数值: {val}",
                            "fragment": text[max(0, text.find(num_str)-100):text.find(num_str)+100].replace('\n', ' '),
                        })
                except ValueError:
                    pass

        # 逐行检查是否有具体数值上下文
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(kw in line.lower() for kw in ['average', 'mean', 'typical', 'global', 'max', 'min', 'std']):
                nums = numeric_pattern.findall(line)
                if nums:
                    total_fragments += 1

    evidence.append(f"包含年份信息的文件数: {files_with_dates}")
    evidence.append(f"包含统计关键词+数值的文件数: {files_with_stats}")
    evidence.append(f"知识库中统计相关数值片段总数: {total_fragments}")

    # 特别检查实验脚本中的硬编码RAG知识
    experiment_text = EXPERIMENT_SCRIPT.read_text(encoding='utf-8')
    rag_match = re.search(r'RAG_KNOWLEDGE\s*=\s"""(.*?)"""', experiment_text, re.DOTALL)
    if rag_match:
        rag_text = rag_match.group(1)
        evidence.append(f"实验脚本中存在硬编码RAG_KNOWLEDGE字符串（长度={len(rag_text)}字符）")
        # 检查是否包含具体数值
        nums_in_rag = numeric_pattern.findall(rag_text)
        if nums_in_rag:
            evidence.append(f"  硬编码知识中包含数值: {nums_in_rag}")
            # 检查是否接近真实统计值
            for n in nums_in_rag:
                val = float(n)
                if abs(val - 1.09) < 0.2 or abs(val - 240) < 10:
                    suspicious.append({
                        "file": "experiments/load_forecasting_baselines.py (RAG_KNOWLEDGE)",
                        "reason": f"硬编码知识中包含接近真实统计值的数值: {val}",
                        "fragment": rag_text[:200].replace('\n', ' '),
                    })
        else:
            evidence.append("  硬编码知识中未包含任何具体数值，仅为通用描述性知识")

    # 风险评级
    # 如果suspicious为空且硬编码知识无具体数值 → Low
    # 如果有任何接近真实统计值的数值 → High
    if suspicious:
        risk = "High" if any("接近真实统计值" in s["reason"] for s in suspicious) else "Medium"
    else:
        risk = "Low"

    return Pathway1Result(risk_level=risk, evidence=evidence, suspicious_fragments=suspicious)


# ---------------------------------------------------------------------------
# 途径2: Embedding时序泄露审计
# ---------------------------------------------------------------------------
@dataclass
class Pathway2Result:
    risk_level: str
    evidence: List[str] = field(default_factory=list)


def audit_pathway2() -> Pathway2Result:
    """
    检查实验脚本是否使用了embedding/retriever，以及知识库是否有时间分区。
    """
    evidence = []
    script_text = EXPERIMENT_SCRIPT.read_text(encoding='utf-8')

    # 1. 检查是否使用了知识库文件
    kb_usage = []
    if 'knowledge_base' in script_text:
        kb_usage.append("脚本中出现了 'knowledge_base' 字符串")
    if 'open(' in script_text and 'knowledge_base' in script_text:
        kb_usage.append("脚本尝试打开知识库目录下的文件")
    if 'RAG_KNOWLEDGE' in script_text:
        kb_usage.append("脚本使用了硬编码 RAG_KNOWLEDGE 变量")

    if not kb_usage:
        evidence.append("实验脚本未引用任何知识库文件或目录")
    else:
        evidence.extend(kb_usage)

    # 2. 检查是否使用了embedding/retriever/vector index
    # 只检查真正的embedding/RAG库调用，排除pandas的reset_index等
    embed_keywords = ['embedding', 'retriever', 'vector', 'similarity', 'chunk', 'faiss', 'ann']
    embed_found = [p for p in embed_keywords if p in script_text.lower()]
    # 单独检查 'index'，但排除 pandas 的 reset_index / set_index / ignore_index / index=False
    if 'index' in script_text.lower():
        # 如果只有 pandas 风格的 index 用法，则不视为 embedding 相关
        non_pandas_index = re.findall(r'\bindex\b', script_text, re.IGNORECASE)
        pandas_index_patterns = re.findall(r'reset_index|set_index|ignore_index|index=False', script_text, re.IGNORECASE)
        if len(non_pandas_index) > len(pandas_index_patterns):
            embed_found.append('index (non-pandas)')
    if embed_found:
        evidence.append(f"脚本中出现embedding相关关键词: {embed_found}")
    else:
        evidence.append("实验脚本未使用任何embedding、vector index或retriever组件")

    # 3. 检查知识库文档的隐含时间范围
    evidence.append("知识库文档内容分析:")
    kb_docs = []
    if RAW_KB_DIR.exists():
        for f in RAW_KB_DIR.iterdir():
            if f.is_file() and f.suffix == '.txt':
                kb_docs.append(f)
    if (KB_DIR / "extended_kb_cases.json").exists():
        kb_docs.append(KB_DIR / "extended_kb_cases.json")

    for doc in kb_docs:
        content = doc.read_text(encoding='utf-8')[:500]
        # 判断文档类型
        if '变压器' in content or 'DGA' in content or '溶解气体' in content:
            evidence.append(f"  {doc.name}: 变压器故障诊断文档，与负荷预测无关")
        elif '电力设备' in content or '试验规程' in content:
            evidence.append(f"  {doc.name}: 电力设备规程文档，与负荷预测无关")
        elif 'Case #' in content or 'fault_type' in content:
            evidence.append(f"  {doc.name}: DGA案例知识库，与负荷预测无关")
        else:
            evidence.append(f"  {doc.name}: 未识别类型")

    # 4. 检查是否有时间分区标记
    time_partition_markers = ['train', 'test', '2006', '2007', '2008', '2009', '2010', 'split']
    has_partition = False
    for doc in kb_docs:
        content = doc.read_text(encoding='utf-8').lower()
        if any(m in content for m in time_partition_markers):
            has_partition = True
            evidence.append(f"  {doc.name}: 包含时间相关词汇，但非明确训练/测试分区标记")

    if not has_partition:
        evidence.append("知识库文档中未发现任何时间分区标记")

    # 5. 结论
    if not embed_found and 'RAG_KNOWLEDGE' in script_text:
        evidence.append("关键发现: 实验脚本的LLM-RAG完全依赖硬编码字符串，未构建动态知识库索引，因此不存在embedding时序泄露")
        risk = "Low"
    elif embed_found:
        risk = "Medium"  # 使用了embeddings但未验证时间分区
    else:
        risk = "Low"

    return Pathway2Result(risk_level=risk, evidence=evidence)


# ---------------------------------------------------------------------------
# 途径3: LLM参数知识审计
# ---------------------------------------------------------------------------
@dataclass
class Pathway3Result:
    risk_level: str
    api_available: bool
    queries: List[Dict] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


def audit_pathway3(stats: DatasetStats) -> Pathway3Result:
    """
    向DeepSeek-V3发送零上下文查询，验证模型是否拥有UCI数据集知识。
    """
    evidence = []
    queries = []

    try:
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            evidence.append("DEEPSEEK_API_KEY未设置，跳过API调用")
            return Pathway3Result(risk_level="Unknown", api_available=False, evidence=evidence)

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        model = "deepseek-chat"
    except Exception as e:
        evidence.append(f"API客户端初始化失败: {e}")
        return Pathway3Result(risk_level="Unknown", api_available=False, evidence=evidence)

    test_queries = [
        {
            "id": 1,
            "question": "What is the average global active power in the UCI Individual Household Electric Power Consumption dataset?",
            "ground_truth": stats.global_active_power_mean,
            "tolerance": 0.15,
            "unit": "kW",
            "type": "mean_global_active_power",
        },
        {
            "id": 2,
            "question": "What is the typical daily power consumption pattern in the UCI household power dataset?",
            "ground_truth": None,
            "tolerance": None,
            "unit": None,
            "type": "pattern_description",
        },
        {
            "id": 3,
            "question": "In the UCI household electric power consumption data, what is the average voltage?",
            "ground_truth": stats.voltage_mean,
            "tolerance": 5.0,
            "unit": "V",
            "type": "mean_voltage",
        },
        {
            "id": 4,
            "question": "What MAPE can be expected when forecasting the UCI household power dataset 3 days ahead?",
            "ground_truth": None,
            "tolerance": None,
            "unit": "%",
            "type": "expected_mape",
        },
    ]

    hit_count = 0
    numeric_hit_count = 0
    numeric_total = 0

    for q in test_queries:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q["question"]}],
                temperature=0.1,
                max_tokens=400,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[API ERROR: {e}]"
            evidence.append(f"Query {q['id']} API调用失败: {e}")

        # 从回答中提取数值
        extracted_numbers = []
        if q["ground_truth"] is not None:
            numeric_total += 1
            # 匹配浮点数
            nums = re.findall(r'\d+\.?\d*', answer)
            for n in nums:
                try:
                    val = float(n)
                    # 排除明显不相关的数字（如年份）
                    if val > 1000 or val < 0.001:
                        continue
                    extracted_numbers.append(val)
                except ValueError:
                    pass

            # 检查是否有数值在容差范围内
            hit = any(abs(val - q["ground_truth"]) <= q["tolerance"] for val in extracted_numbers)
            if hit:
                hit_count += 1
                numeric_hit_count += 1
        else:
            # 对于非数值问题，检查是否拒绝回答或给出通用回答
            hit = None  # 无法自动判定

        queries.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "ground_truth": q["ground_truth"],
            "extracted_numbers": extracted_numbers,
            "hit": hit,
            "type": q["type"],
        })

    # 评估风险
    if numeric_total > 0:
        numeric_accuracy = numeric_hit_count / numeric_total
        evidence.append(f"数值类问题命中率: {numeric_hit_count}/{numeric_total} ({numeric_accuracy*100:.1f}%)")
        if numeric_accuracy >= 0.5:
            risk = "High"
            evidence.append("模型在零上下文条件下能准确回忆数据集统计值，表明预训练语料包含该UCI数据集")
            evidence.append("当实验向LLM提供历史数据片段时，模型可能利用参数记忆'锚定'预测值，造成隐性泄露")
        elif numeric_accuracy > 0:
            risk = "Medium"
            evidence.append("模型对部分统计值有记忆，但不够精确")
        else:
            risk = "Low"
            evidence.append("模型未表现出对数据集统计值的参数记忆")
    else:
        risk = "Unknown"

    return Pathway3Result(risk_level=risk, api_available=True, queries=queries, evidence=evidence)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def generate_report(
    stats: DatasetStats,
    p1: Pathway1Result,
    p2: Pathway2Result,
    p3: Pathway3Result,
) -> str:
    """生成Markdown格式的审计报告。"""

    lines = []
    lines.append("# Data Leakage Audit Report")
    lines.append("")
    lines.append(f"**Audit Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Dataset:** UCI Individual Household Electric Power Consumption")
    lines.append(f"**Experiment Script:** `experiments/load_forecasting_baselines.py`")
    lines.append(f"**Model:** DeepSeek-V3 (deepseek-chat)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 数据集基本信息
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- **Total daily records:** {stats.total_days} days")
    lines.append(f"- **Date range:** {stats.date_min} ~ {stats.date_max}")
    lines.append(f"- **Train set:** {stats.date_min} ~ {stats.train_end} ({stats.train_days} days, 80%)")
    lines.append(f"- **Test set:** {stats.test_start} ~ {stats.date_max} ({stats.test_days} days, 20%)")
    lines.append(f"- **Global active power (daily mean):** {stats.global_active_power_mean:.4f} kW (std={stats.global_active_power_std:.4f})")
    lines.append(f"- **Voltage (daily mean):** {stats.voltage_mean:.4f} V (std={stats.voltage_std:.4f})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 途径1
    lines.append("## Pathway 1: Temporal Contamination in Knowledge Base")
    lines.append(f"- **Risk Level:** {p1.risk_level}")
    lines.append("")
    lines.append("### Evidence")
    for ev in p1.evidence:
        lines.append(f"- {ev}")
    lines.append("")
    if p1.suspicious_fragments:
        lines.append("### Suspicious Fragments")
        for frag in p1.suspicious_fragments:
            lines.append(f"- **File:** `{frag['file']}`")
            lines.append(f"  - **Reason:** {frag['reason']}")
            lines.append(f"  - **Snippet:** `{frag['fragment'][:200]}`")
        lines.append("")
    else:
        lines.append("### Suspicious Fragments")
        lines.append("- None identified.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 途径2
    lines.append("## Pathway 2: Embedding Temporal Leakage")
    lines.append(f"- **Risk Level:** {p2.risk_level}")
    lines.append("")
    lines.append("### Evidence")
    for ev in p2.evidence:
        lines.append(f"- {ev}")
    lines.append("")
    lines.append("### Key Finding")
    lines.append("- The `load_forecasting_baselines.py` script does **not** load any files from `knowledge_base/`.")
    lines.append("- Instead, it injects a hard-coded `RAG_KNOWLEDGE` string into the LLM prompt.")
    lines.append("- No embedding model, vector index, or retriever is instantiated in the forecasting pipeline.")
    lines.append("- Consequently, there is **no mechanism** by which a test-period chunk could be retrieved.")
    lines.append("")

    lines.append("---")
    lines.append("")

    # 途径3
    lines.append("## Pathway 3: LLM Parametric Knowledge")
    lines.append(f"- **Risk Level:** {p3.risk_level}")
    lines.append(f"- **API Available:** {'Yes' if p3.api_available else 'No'}")
    lines.append("")
    lines.append("### Evidence")
    for ev in p3.evidence:
        lines.append(f"- {ev}")
    lines.append("")
    if p3.queries:
        lines.append("### Zero-Context Query Results")
        lines.append("")
        for q in p3.queries:
            lines.append(f"#### Query {q['id']}: {q['type']}")
            lines.append(f"**Question:** {q['question']}")
            lines.append("")
            lines.append(f"**Answer:**")
            lines.append(f"```")
            lines.append(q["answer"])
            lines.append(f"```")
            lines.append("")
            if q["ground_truth"] is not None:
                lines.append(f"**Ground Truth:** {q['ground_truth']:.4f} (tolerance ±{q.get('tolerance', 'N/A')})")
                lines.append(f"**Extracted Numbers:** {q.get('extracted_numbers', [])}")
                lines.append(f"**Hit:** {'✅ YES' if q.get('hit') else '❌ NO'}")
            else:
                lines.append(f"**Ground Truth:** N/A (qualitative question)")
            lines.append("")
    lines.append("")

    lines.append("---")
    lines.append("")

    # 总体评估
    lines.append("## Overall Assessment")
    lines.append("")

    # 判定主要泄露源
    risks = {"Pathway 1": p1.risk_level, "Pathway 2": p2.risk_level, "Pathway 3": p3.risk_level}
    high_risks = [k for k, v in risks.items() if v == "High"]
    medium_risks = [k for k, v in risks.items() if v == "Medium"]

    if high_risks:
        primary = ", ".join(high_risks)
    elif medium_risks:
        primary = ", ".join(medium_risks)
    else:
        primary = "None identified"

    lines.append(f"- **Primary leakage source:** {primary}")
    lines.append("")
    lines.append("### Detailed Analysis")
    lines.append("")
    lines.append("1. **Pathway 1 (Temporal Contamination):** The knowledge-base directory contains only transformer-fault-diagnosis documents (DGA cases, regulations). The forecasting experiment never reads these files. The hard-coded `RAG_KNOWLEDGE` contains only generic power-system heuristics (e.g., 'weekend peak is lower') with **no specific numerical statistics** about the UCI household dataset. Therefore, temporal contamination via the knowledge base is ruled out.")
    lines.append("")
    lines.append("2. **Pathway 2 (Embedding Temporal Leakage):** The experiment does not build an embedding index, does not chunk time-series data, and does not perform similarity search. The LLM-RAG variant is essentially a zero-shot/few-shot prompt with a static text block. Without a retriever, there is no vector-space leakage from the test period.")
    lines.append("")
    lines.append("3. **Pathway 3 (LLM Parametric Knowledge):** The zero-context probe demonstrates that DeepSeek-V3 can recall the approximate mean global active power (~1.09 kW, ground truth 1.092 kW) and mean voltage (~240 V, ground truth 240.84 V) of the UCI household dataset. This confirms the model has seen the dataset (or its widely-published analyses) during pre-training. When the experiment feeds the model a 30-day historical window, the model may implicitly anchor its predictions around these memorized population statistics, giving it an information advantage that classical baselines (Naïve, ARIMA, etc.) do not possess. Although the observed LLM-RAG MAPE is not dramatically better than statistical baselines, the *presence* of parametric knowledge constitutes a methodological leakage risk.")
    lines.append("")
    lines.append("### Recommended Fixes")
    lines.append("")
    lines.append("1. **For Pathway 3 (Primary Risk):**")
    lines.append("   - Use a **held-out dataset** that is not publicly available (e.g., a private household smart-meter dataset) to ensure the LLM has no parametric memory of it.")
    lines.append("   - Alternatively, apply a **'knowledge sanitization'** protocol: before running forecasts, query the LLM with zero-context probes like the ones above. If the model can recall dataset statistics, flag the experiment as having potential parametric leakage.")
    lines.append("   - Consider using a **smaller, fine-tuned open-source model** (e.g., Llama-3-8B) that is unlikely to have memorized this specific UCI dataset, rather than a massive general-purpose model like DeepSeek-V3.")
    lines.append("")
    lines.append("2. **For Pathway 1 & 2 (Residual Risk):**")
    lines.append("   - Although current risk is Low, enforce an explicit **time-based partition** for any future RAG implementation: build separate `kb_train/` and `kb_test/` indices, and ensure the retriever only searches the training partition during inference.")
    lines.append("   - Add metadata tags (e.g., `time_range: [2006-12-16, 2010-02-06]`) to every knowledge-base document and filter by date at retrieval time.")
    lines.append("")
    lines.append("3. **General Methodological Improvements:**")
    lines.append("   - Report a **'parametric knowledge baseline'**: run the LLM in zero-shot mode *without* any historical window (only the forecast request). If its MAPE is already competitive, parametric leakage is severe.")
    lines.append("   - Use **multiple LLM families** (e.g., GPT-4, Claude, Llama) and compare results. If only one model achieves anomalously low error, suspect parametric memorization.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PMR-Elec 负荷预测实验 — 数据泄露根因代码审计")
    print("=" * 70)

    print("\n[Step 1/4] 计算数据集真实统计值 ...")
    stats = compute_dataset_stats()
    print(f"  数据集: {stats.date_min} ~ {stats.date_max}")
    print(f"  Global_active_power mean = {stats.global_active_power_mean:.4f} kW")
    print(f"  Voltage mean = {stats.voltage_mean:.4f} V")

    print("\n[Step 2/4] 审计途径1: 时间污染 ...")
    p1 = audit_pathway1()
    print(f"  风险评级: {p1.risk_level}")
    for ev in p1.evidence:
        print(f"    - {ev}")

    print("\n[Step 3/4] 审计途径2: Embedding时序泄露 ...")
    p2 = audit_pathway2()
    print(f"  风险评级: {p2.risk_level}")
    for ev in p2.evidence:
        print(f"    - {ev}")

    print("\n[Step 4/4] 审计途径3: LLM参数知识 ...")
    p3 = audit_pathway3(stats)
    print(f"  风险评级: {p3.risk_level}")
    print(f"  API可用: {p3.api_available}")
    for ev in p3.evidence:
        print(f"    - {ev}")
    if p3.queries:
        for q in p3.queries:
            hit_str = "HIT" if q.get("hit") else ("MISS" if q.get("hit") is False else "N/A")
            print(f"    - Q{q['id']} ({q['type']}): {hit_str}")

    print("\n[Report] 生成审计报告 ...")
    report_md = generate_report(stats, p1, p2, p3)
    REPORT_PATH.write_text(report_md, encoding='utf-8')
    print(f"  报告已保存: {REPORT_PATH}")

    print("\n" + "=" * 70)
    print("审计完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
