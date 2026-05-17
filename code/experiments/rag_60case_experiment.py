#!/usr/bin/env python3
"""
RAG 60-Case Knowledge Base Diagnostic Experiment
=================================================
Compare downstream diagnosis performance:
  Config A: 8-document text-based KB (rule descriptions)
  Config B: 60-case structured KB (retrieval-augmented)

Test set: stratified random subset, 10 samples per class × 6 classes = 60 samples
Embedding: Ollama bge-m3:latest (1024-dim)
LLM: DeepSeek-V3 (deepseek-chat), temperature=0

Outputs:
  - experiments/results/rag_60case_results.csv
  - experiments/results/rag_60case_summary.md
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dga_transformer_data.csv"
KB_PATH = PROJECT_ROOT / "knowledge_base" / "extended_kb_cases.json"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 6 fault classes in the real dataset
CLASS_NAMES = [
    "Normal",
    "Partial Discharge",
    "Low Energy Discharge",
    "High Energy Discharge",
    "Low Temperature Thermal",
    "High Temperature Thermal",
]

CLASS_NAME_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

# Map underscore names in KB/CSV to display names
UNDERSCORE_TO_DISPLAY = {
    "Normal": "Normal",
    "Partial_Discharge": "Partial Discharge",
    "Low_Energy_Discharge": "Low Energy Discharge",
    "High_Energy_Discharge": "High Energy Discharge",
    "Low_Temp_Thermal": "Low Temperature Thermal",
    "High_Temp_Thermal": "High Temperature Thermal",
}

DISPLAY_TO_UNDERSCORE = {v: k for k, v in UNDERSCORE_TO_DISPLAY.items()}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 8-Document KB (Config A) — hard-coded rule-based text descriptions
# Adapted from v2_experiment_main.py for the 6-class dataset
# ---------------------------------------------------------------------------
EIGHT_DOC_KB = [
    """【Fault Type: Normal】Normal state: All gas concentrations are below attention values. Total hydrocarbons < 100 μL/L (110 kV) or < 150 μL/L (220 kV+), C2H2 < 1 μL/L (220 kV+) or < 5 μL/L (110 kV), H2 < 150 μL/L. Ratio encoding is usually 0,0,0. No characteristic gases are produced.""",

    """【Fault Type: Partial Discharge】Partial Discharge: H2 is significantly elevated (usually > 100 μL/L), CH4 is relatively low, and C2H2 is extremely low (< 1 μL/L). Characteristic ratio: R2 > 0.1 and R1 < 0.1. The main characteristic gas is H2, with CH4 as the primary hydrocarbon component.""",

    """【Fault Type: Low Energy Discharge】Low Energy Discharge: C2H2 is moderately elevated (5–30 μL/L), C2H4 is moderately elevated (30–150 μL/L). Characteristic ratio: 0.1 < R1 < 3. The C2H2/C2H4 ratio is between 0.1 and 3; C2H2 content is the key indicator.""",

    """【Fault Type: High Energy Discharge】High Energy Discharge: C2H2 is significantly elevated (> 30 μL/L), C2H4 is significantly elevated (> 100 μL/L). Characteristic ratio: R1 > 3. The C2H2/C2H4 ratio > 3, and C2H2 is much higher than in low-energy discharge. May involve arc discharge and is the most severe discharge fault.""",

    """【Fault Type: Low Temperature Thermal】Low Temperature Thermal (< 300°C): CH4 is significantly elevated, C2H6 content is relatively high. Characteristic ratio: R2 > 1, R1 < 0.1. Main characteristic gases are CH4 and C2H6, temperature approximately 150–300°C. Commonly caused by poor contact, eddy currents, etc.""",

    """【Fault Type: High Temperature Thermal】High Temperature Thermal (> 700°C): C2H4 is significantly elevated (> 300 μL/L), CH4 is moderate. Characteristic ratio: R2 > 1, R1 > 0.1 but C2H2 remains low. Temperature > 700°C; C2H4 becomes the dominant characteristic gas. Commonly caused by severe overload, winding short circuits, etc.""",

    """【Standard: DL/T 722-2014】Core guidelines of DL/T 722-2014 "Guide to the Analysis and Diagnosis of Dissolved Gases in Transformer Oil": 1. Total hydrocarbon attention value: 150 μL/L (220 kV and above), 100 μL/L (110 kV). 2. Acetylene (C2H2) attention value: 1 μL/L (220 kV and above), 5 μL/L (110 kV). 3. Hydrogen (H2) attention value: 150 μL/L (220 kV and above). 4. Three-ratio encoding rules: C2H2/C2H4 (<0.1→0, 0.1–3→1, >3→2); CH4/H2 (<0.1→0, 0.1–1→1, >1→2); C2H2/C2H6 (<0.1→0, 0.1–3→1, >3→2). 5. Gas production rate is also an important indicator: absolute gas production rate > 6 mL/d (open type) or > 12 mL/d (sealed type) requires attention.""",

    """【Rule: Three-Ratio Encoding Table】Three-ratio encoding and fault type correspondence table: Code (0,0,1) → Normal; Code (0,1,0) → Partial Discharge; Code (1,0,0) or (1,0,1) → Low Energy Discharge; Code (1,0,2), (1,1,0), (1,1,1), (1,1,2) → High Energy Discharge; Code (0,0,2) → Low Temperature Thermal; Code (0,2,0), (0,2,1), (0,2,2) → High Temperature Thermal.""",
]

# ---------------------------------------------------------------------------
# Ollama Embedder
# ---------------------------------------------------------------------------
class OllamaEmbedder:
    """Local Ollama embedding via HTTP API."""

    def __init__(self, model_name: str = "bge-m3:latest"):
        self.model_name = model_name
        import urllib.request
        self._request = urllib.request

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in tqdm(texts, desc="Embedding", leave=False):
            req = self._request.Request(
                "http://localhost:11434/api/embeddings",
                data=json.dumps({"model": self.model_name, "prompt": str(text)}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self._request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())
                emb = np.array(result["embedding"], dtype=np.float32)
                embeddings.append(emb)
        emb_matrix = np.stack(embeddings)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        return emb_matrix / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# LLM Client (DeepSeek-V3)
# ---------------------------------------------------------------------------
@dataclass
class LLMClient:
    model_name: str = "deepseek-chat"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 3

    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)

    def __post_init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai library not installed") from e

        if self.api_key is None:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

    def call(self, messages: List[Dict[str, str]], max_tokens: int = 100, timeout: float = 60.0) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                content = resp.choices[0].message.content.strip()
                self.total_prompt_tokens += resp.usage.prompt_tokens
                self.total_completion_tokens += resp.usage.completion_tokens
                # DeepSeek-V3 pricing (per 1M tokens, RMB): input 1元, output 2元
                self.total_cost += (
                    resp.usage.prompt_tokens * 1.0 / 1e6
                    + resp.usage.completion_tokens * 2.0 / 1e6
                )
                return content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[LLM] API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_config_a_prompt(sample_text: str) -> str:
    """Config A: 8-document rule-based KB."""
    docs_text = "\n\n".join(EIGHT_DOC_KB)
    return (
        "You are a transformer fault diagnosis expert. "
        "Below is the rule-based knowledge base for DGA fault diagnosis:\n\n"
        f"{docs_text}\n\n"
        "Given the new DGA record:\n"
        f"{sample_text}\n\n"
        "What is the fault type? Answer with exactly one class from: "
        "Normal, Partial Discharge, Low Energy Discharge, High Energy Discharge, "
        "Low Temperature Thermal, High Temperature Thermal. "
        "Reply with only the class name, no explanation."
    )


def build_config_b_prompt(sample_text: str, top3_cases: List[Dict[str, Any]]) -> str:
    """Config B: 60-case KB with Top-3 retrieved cases."""
    case_lines = []
    for i, case in enumerate(top3_cases, 1):
        case_lines.append(f"[{i}] {case['text']}")
    cases_text = "\n".join(case_lines)
    return (
        "You are a transformer fault diagnosis expert. "
        "Below are the 3 most similar historical cases retrieved from the knowledge base:\n\n"
        f"{cases_text}\n\n"
        "Given the new DGA record:\n"
        f"{sample_text}\n\n"
        "What is the fault type? Answer with exactly one class from: "
        "Normal, Partial Discharge, Low Energy Discharge, High Energy Discharge, "
        "Low Temperature Thermal, High Temperature Thermal. "
        "Reply with only the class name, no explanation."
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_real_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure consistent class names
    df["Fault_Display"] = df["Fault"].map(lambda x: UNDERSCORE_TO_DISPLAY.get(x, x))
    return df


def sample_stratified_test_set(df: pd.DataFrame, n_per_class: int = 10) -> pd.DataFrame:
    """Stratified random sampling: n_per_class samples for each class."""
    samples = []
    for fault in CLASS_NAMES:
        subset = df[df["Fault_Display"] == fault]
        if len(subset) < n_per_class:
            logger.warning(f"Class {fault} has only {len(subset)} samples (requested {n_per_class})")
            selected = subset
        else:
            selected = subset.sample(n=n_per_class, random_state=SEED)
        samples.append(selected)
    return pd.concat(samples).sample(frac=1, random_state=SEED).reset_index(drop=True)


def row_to_text(row: pd.Series) -> str:
    """Compact structured text for embedding and prompt."""
    return (
        f"H2={row['H2']:.2f}, CH4={row['CH4']:.2f}, C2H6={row['C2H6']:.2f}, "
        f"C2H4={row['C2H4']:.2f}, C2H2={row['C2H2']:.2f}, "
        f"Total_HC={row['Total_HC']:.2f}, CO={row['CO']:.2f}, CO2={row['CO2']:.2f}"
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
def parse_llm_response(text: str) -> str:
    """Extract exact class name from LLM response."""
    text = text.strip()
    # Direct match
    for cls in CLASS_NAMES:
        if text.lower() == cls.lower():
            return cls
    # Partial match (contains class name)
    for cls in CLASS_NAMES:
        if cls.lower() in text.lower():
            return cls
    # Handle underscore variants (both short and full forms)
    underscore_mappings = {
        "normal": "Normal",
        "partial_discharge": "Partial Discharge",
        "low_energy_discharge": "Low Energy Discharge",
        "high_energy_discharge": "High Energy Discharge",
        "low_temp_thermal": "Low Temperature Thermal",
        "low_temperature_thermal": "Low Temperature Thermal",
        "high_temp_thermal": "High Temperature Thermal",
        "high_temperature_thermal": "High Temperature Thermal",
    }
    lowered = text.lower().replace(" ", "_")
    for under, disp in underscore_mappings.items():
        if under in lowered:
            return disp
    # Try to find numeric labels 0-5
    m = re.search(r"\b([0-5])\b", text)
    if m:
        return CLASS_NAMES[int(m.group(1))]
    return "Unknown"


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d) -> (n, m)"""
    return a @ b.T


def retrieve_topk(query_emb: np.ndarray, kb_emb: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and similarities."""
    sims = cosine_similarity(query_emb, kb_emb)
    topk_idx = np.argsort(-sims, axis=1)[:, :k]
    topk_sims = np.take_along_axis(sims, topk_idx, axis=1)
    return topk_idx, topk_sims


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("RAG 60-Case Knowledge Base Diagnostic Experiment")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("[Step 1/7] Loading real data and KB...")
    df = load_real_data(DATA_PATH)
    logger.info(f"Real data: {len(df)} samples, classes: {df['Fault_Display'].value_counts().to_dict()}")

    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    cases = kb_data["cases"]
    logger.info(f"KB loaded: {len(cases)} cases")

    # -----------------------------------------------------------------------
    # 2. Stratified sampling
    # -----------------------------------------------------------------------
    logger.info("[Step 2/7] Stratified sampling: 10 per class -> 60 test samples")
    test_df = sample_stratified_test_set(df, n_per_class=10)
    logger.info(f"Test set: {len(test_df)} samples")
    logger.info(f"Test distribution: {test_df['Fault_Display'].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # 3/4. Load or generate embeddings
    # -----------------------------------------------------------------------
    kb_embs_path = RESULTS_DIR / "kb_embs.npy"
    test_embs_path = RESULTS_DIR / "test_embs.npy"
    if kb_embs_path.exists() and test_embs_path.exists():
        logger.info("[Step 3-4/7] Loading pre-computed embeddings...")
        kb_embs = np.load(kb_embs_path)
        test_embs = np.load(test_embs_path)
    else:
        logger.info("[Step 3-4/7] Pre-computed embeddings not found. Generating via Ollama (this may take ~5 min)...")
        embedder = OllamaEmbedder(model_name="bge-m3:latest")
        kb_texts = [case["text"] for case in cases]
        kb_embs = embedder.encode(kb_texts)
        np.save(kb_embs_path, kb_embs)
        test_texts = [row_to_text(row) for _, row in test_df.iterrows()]
        test_embs = embedder.encode(test_texts)
        np.save(test_embs_path, test_embs)
    logger.info(f"KB embeddings shape: {kb_embs.shape}")
    logger.info(f"Test embeddings shape: {test_embs.shape}")

    # -----------------------------------------------------------------------
    # 5. Retrieval: Top-3 for each test sample
    # -----------------------------------------------------------------------
    logger.info("[Step 5/7] Computing cosine similarity and Top-3 retrieval...")
    topk_idx, topk_sims = retrieve_topk(test_embs, kb_embs, k=3)

    # Evaluate retrieval quality
    top1_correct = 0
    avg_sim_sum = 0.0
    retrieval_details = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        true_class = row["Fault_Display"]
        top3 = []
        for rank in range(3):
            case_idx = int(topk_idx[i, rank])
            sim = float(topk_sims[i, rank])
            case = cases[case_idx]
            case_class = UNDERSCORE_TO_DISPLAY[case["fault_type"]]
            top3.append({
                "rank": rank + 1,
                "case_id": case["case_number"],
                "case_class": case_class,
                "similarity": round(sim, 4),
                "text": case["text"],
            })
        avg_sim_sum += float(topk_sims[i, :3].mean())
        if top3[0]["case_class"] == true_class:
            top1_correct += 1
        retrieval_details.append({
            "sample_idx": i,
            "true_class": true_class,
            "top3": top3,
        })

    top1_acc = top1_correct / len(test_df)
    avg_sim = avg_sim_sum / len(test_df)
    logger.info(f"Top-1 retrieval accuracy: {top1_acc:.4f}")
    logger.info(f"Average Top-3 cosine similarity: {avg_sim:.4f}")

    # -----------------------------------------------------------------------
    # 6. LLM diagnosis: Config A and Config B (with resume support)
    # -----------------------------------------------------------------------
    logger.info("[Step 6/7] Calling DeepSeek-V3 for diagnosis (Config A + Config B)...")
    llm = LLMClient(model_name="deepseek-chat", temperature=0.0)

    checkpoint_path = RESULTS_DIR / "rag_60case_checkpoint.json"
    completed_indices = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        completed_indices = set(checkpoint.get("completed", []))
        logger.info(f"[Resume] Found checkpoint with {len(completed_indices)} completed samples")

    results = []
    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Diagnosis")):
        if i in completed_indices:
            # Load from checkpoint if available
            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                results.append(checkpoint["results"][i])
            continue

        sample_text = row_to_text(row)
        true_class = row["Fault_Display"]

        # --- Config A: 8-document KB ---
        prompt_a = build_config_a_prompt(sample_text)
        messages_a = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_a},
        ]
        resp_a = llm.call(messages_a, max_tokens=50)
        pred_a = parse_llm_response(resp_a)

        # --- Config B: 60-case KB Top-3 ---
        top3_cases = retrieval_details[i]["top3"]
        prompt_b = build_config_b_prompt(sample_text, top3_cases)
        messages_b = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_b},
        ]
        resp_b = llm.call(messages_b, max_tokens=50)
        pred_b = parse_llm_response(resp_b)

        result = {
            "sample_idx": i,
            "true_class": true_class,
            "config_a_pred": pred_a,
            "config_a_correct": int(pred_a == true_class),
            "config_b_pred": pred_b,
            "config_b_correct": int(pred_b == true_class),
            "top1_case_id": top3_cases[0]["case_id"],
            "top1_case_class": top3_cases[0]["case_class"],
            "top1_similarity": top3_cases[0]["similarity"],
            "top1_retrieval_correct": int(top3_cases[0]["case_class"] == true_class),
            "top2_case_id": top3_cases[1]["case_id"],
            "top2_case_class": top3_cases[1]["case_class"],
            "top2_similarity": top3_cases[1]["similarity"],
            "top3_case_id": top3_cases[2]["case_id"],
            "top3_case_class": top3_cases[2]["case_class"],
            "top3_similarity": top3_cases[2]["similarity"],
            "avg_top3_sim": round(sum(c["similarity"] for c in top3_cases) / 3, 4),
            "raw_resp_a": resp_a,
            "raw_resp_b": resp_b,
        }
        results.append(result)

        # Save checkpoint
        checkpoint = {
            "completed": list(range(i + 1)),
            "results": results,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        # Rate limit protection
        time.sleep(0.3)

    logger.info(f"[LLM] Total API calls: {len(test_df) * 2}")
    logger.info(f"[LLM] Estimated cost: ¥{llm.total_cost:.4f} (RMB)")

    # -----------------------------------------------------------------------
    # 7. Evaluation & Save
    # -----------------------------------------------------------------------
    logger.info("[Step 7/7] Evaluating and saving results...")

    df_results = pd.DataFrame(results)
    results_csv = RESULTS_DIR / "rag_60case_results.csv"
    df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Detailed results saved to: {results_csv}")

    # Overall metrics
    acc_a = accuracy_score(df_results["true_class"], df_results["config_a_pred"])
    acc_b = accuracy_score(df_results["true_class"], df_results["config_b_pred"])

    # Per-class F1
    y_true = df_results["true_class"].values
    y_pred_a = df_results["config_a_pred"].values
    y_pred_b = df_results["config_b_pred"].values

    labels = CLASS_NAMES
    prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(y_true, y_pred_a, labels=labels, zero_division=0)
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_true, y_pred_b, labels=labels, zero_division=0)

    # Macro F1
    macro_f1_a = np.mean(f1_a)
    macro_f1_b = np.mean(f1_b)

    # Build summary table rows
    summary_rows = []
    for idx, cls in enumerate(labels):
        summary_rows.append({
            "Metric": f"F1 ({cls})",
            "Config A (8-doc KB)": round(f1_a[idx], 4),
            "Config B (60-case KB)": round(f1_b[idx], 4),
            "Delta": round(f1_b[idx] - f1_a[idx], 4),
        })
    summary_rows.append({
        "Metric": "Accuracy",
        "Config A (8-doc KB)": round(acc_a, 4),
        "Config B (60-case KB)": round(acc_b, 4),
        "Delta": round(acc_b - acc_a, 4),
    })
    summary_rows.append({
        "Metric": "Macro F1",
        "Config A (8-doc KB)": round(macro_f1_a, 4),
        "Config B (60-case KB)": round(macro_f1_b, 4),
        "Delta": round(macro_f1_b - macro_f1_a, 4),
    })
    summary_rows.append({
        "Metric": "Top-1 Retrieval Acc",
        "Config A (8-doc KB)": "N/A",
        "Config B (60-case KB)": round(top1_acc, 4),
        "Delta": "N/A",
    })
    summary_rows.append({
        "Metric": "Avg Top-3 Cosine Sim",
        "Config A (8-doc KB)": "N/A",
        "Config B (60-case KB)": round(avg_sim, 4),
        "Delta": "N/A",
    })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "rag_60case_summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Confusion matrices
    cm_a = confusion_matrix(y_true, y_pred_a, labels=labels)
    cm_b = confusion_matrix(y_true, y_pred_b, labels=labels)

    # Classification reports
    report_a = classification_report(y_true, y_pred_a, labels=labels, zero_division=0)
    report_b = classification_report(y_true, y_pred_b, labels=labels, zero_division=0)

    # Generate markdown summary
    md_lines = [
        "# RAG 60-Case Knowledge Base Diagnostic Experiment — Summary",
        "",
        "## Experiment Setup",
        "",
        "| Parameter | Value |",
        "|---|---|",
        "| Test samples | 60 (10 per class × 6 classes) |",
        "| Embedding model | Ollama `bge-m3:latest` (1024-dim) |",
        "| LLM | DeepSeek-V3 (`deepseek-chat`), temperature=0 |",
        "| Retrieval | Cosine similarity, Top-3 |",
        "| API calls | 120 (60 × 2 configs) |",
        f"| Estimated cost | ¥{llm.total_cost:.4f} RMB |",
        "",
        "## Comparison Table",
        "",
        df_summary.to_markdown(index=False),
        "",
        "## Detailed Metrics",
        "",
        "### Config A: 8-Document Rule-Based KB",
        "",
        f"- **Accuracy**: {acc_a:.4f}",
        f"- **Macro F1**: {macro_f1_a:.4f}",
        "",
        "```",
        report_a,
        "```",
        "",
        "### Config B: 60-Case Retrieval-Augmented KB",
        "",
        f"- **Accuracy**: {acc_b:.4f}",
        f"- **Macro F1**: {macro_f1_b:.4f}",
        f"- **Top-1 Retrieval Accuracy**: {top1_acc:.4f}",
        f"- **Average Top-3 Cosine Similarity**: {avg_sim:.4f}",
        "",
        "```",
        report_b,
        "```",
        "",
        "## Confusion Matrices",
        "",
        "### Config A",
        "",
        "| True \\ Pred | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for i, cls in enumerate(labels):
        row = [cls] + [str(cm_a[i, j]) for j in range(len(labels))]
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines += [
        "",
        "### Config B",
        "",
        "| True \\ Pred | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for i, cls in enumerate(labels):
        row = [cls] + [str(cm_b[i, j]) for j in range(len(labels))]
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines += [
        "",
        "## Brief Analysis",
        "",
    ]

    # Auto-generate brief analysis
    delta_acc = acc_b - acc_a
    delta_macro = macro_f1_b - macro_f1_a
    if delta_acc > 0.05:
        acc_comment = "Config B (60-case KB) achieves notably higher accuracy than Config A."
    elif delta_acc < -0.05:
        acc_comment = "Config B (60-case KB) performs notably worse than Config A."
    else:
        acc_comment = "The two configurations show comparable overall accuracy."

    if top1_acc < 0.6:
        retrieval_comment = "Retrieval quality is relatively low, suggesting the 60 cases may not be sufficiently representative or separable in embedding space."
    elif top1_acc < 0.8:
        retrieval_comment = "Retrieval quality is moderate; some cases are confused across classes."
    else:
        retrieval_comment = "Retrieval quality is high, with most Top-1 matches belonging to the correct fault class."

    md_lines.append(
        f"1. **Overall Accuracy**: Config A = {acc_a:.4f}, Config B = {acc_b:.4f} (Δ = {delta_acc:+.4f}). {acc_comment}"
    )
    md_lines.append(
        f"2. **Macro F1**: Config A = {macro_f1_a:.4f}, Config B = {macro_f1_b:.4f} (Δ = {delta_macro:+.4f})."
    )
    md_lines.append(
        f"3. **Retrieval Quality**: Top-1 accuracy = {top1_acc:.4f}; avg Top-3 cosine similarity = {avg_sim:.4f}. {retrieval_comment}"
    )
    md_lines.append(
        "4. **Implications**: If Config B does not significantly outperform Config A, this suggests that simply increasing the number of structured cases in the KB does not guarantee better downstream diagnosis performance. Possible reasons include: (a) case conflicts within the same class introducing noise, (b) embedding-based retrieval capturing superficial similarity rather than diagnostic-relevant patterns, and (c) the LLM already possessing sufficient domain knowledge from the 8-document rules, making additional case examples redundant or even distracting."
    )
    md_lines.append(
        "5. **Honest Report**: This is a negative-finding experiment. Regardless of whether the 60-case KB improves results, the data is reported as-is."
    )
    md_lines.append("")

    summary_md = RESULTS_DIR / "rag_60case_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Summary report saved to: {summary_md}")

    logger.info("=" * 60)
    logger.info("Experiment completed!")
    logger.info(f"  Config A Accuracy: {acc_a:.4f}  |  Config B Accuracy: {acc_b:.4f}")
    logger.info(f"  Config A Macro F1: {macro_f1_a:.4f}  |  Config B Macro F1: {macro_f1_b:.4f}")
    logger.info(f"  Top-1 Retrieval Acc: {top1_acc:.4f}  |  Avg Top-3 Sim: {avg_sim:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
