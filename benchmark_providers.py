#!/usr/bin/env python3
"""
Cross-Provider Benchmark
=========================
Run the same papers through all available LLM providers and compare:
  - Accuracy (if gold standard available)
  - Inter-provider agreement (Cohen's kappa)
  - Cost per 1K papers
  - Latency (median ms/call)
  - Token usage

Usage:
    python benchmark_providers.py --n 100
    python benchmark_providers.py --n 200 --providers openai,anthropic --gold validation_sample.csv
    python benchmark_providers.py --providers openai,google --tasks screen,classify

Output: stats_output/provider_comparison.json + provider_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config import BENCHMARK_SAMPLE_SIZE, LLM_TEMPERATURE
from llm_providers import LLMProvider, DEFAULT_MODELS
from llm_validation import cohens_kappa, kappa_interpretation, wilson_ci

# ── Paths ─────────────────────────────────────────────────────────────
INPUT_FILE = Path(__file__).parent / "results_curated" / "all_filtered.csv"
CACHE_DIR = Path(__file__).parent / "llm_cache"
OUTPUT_DIR = Path(__file__).parent / "stats_output"
OUTPUT_JSON = OUTPUT_DIR / "provider_comparison.json"
OUTPUT_CSV = OUTPUT_DIR / "provider_comparison.csv"


# ═══════════════════════════════════════════════════════════════════════
# PROVIDER DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _detect_available_providers() -> list[dict]:
    """
    Check which LLM providers have valid API keys or are reachable.

    Returns list of dicts: [{"provider": str, "model": str, "method": str}, ...]
    """
    available = []

    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        available.append({
            "provider": "openai",
            "model": DEFAULT_MODELS["openai"],
            "method": "OPENAI_API_KEY env var",
        })

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append({
            "provider": "anthropic",
            "model": DEFAULT_MODELS["anthropic"],
            "method": "ANTHROPIC_API_KEY env var",
        })

    # Google
    if os.environ.get("GOOGLE_API_KEY"):
        available.append({
            "provider": "google",
            "model": DEFAULT_MODELS["google"],
            "method": "GOOGLE_API_KEY env var",
        })

    # Ollama (check HTTP endpoint)
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            available.append({
                "provider": "ollama",
                "model": DEFAULT_MODELS["ollama"],
                "method": "localhost:11434",
            })
    except Exception:
        pass

    return available


# ═══════════════════════════════════════════════════════════════════════
# SINGLE-PROVIDER RUN
# ═══════════════════════════════════════════════════════════════════════

def _run_single_provider(
    provider_name: str,
    model: str,
    papers: pd.DataFrame,
    tasks: list[str],
) -> dict:
    """
    Run all requested tasks for a single provider on the sample papers.

    Returns dict with screening decisions, latencies, token counts.
    """
    print(f"\n  Running {provider_name}/{model}...")

    llm = LLMProvider(
        provider=provider_name,
        model=model,
        temperature=LLM_TEMPERATURE,
    )

    result = {
        "provider": provider_name,
        "model": model,
        "screening": {},
        "latencies_ms": [],
        "errors": 0,
    }

    # ── Screening ────────────────────────────────────────────────────
    if "screen" in tasks:
        from llm_schemas import ScreeningResult
        from llm_screening import build_full_system_prompt, build_user_prompt

        system = build_full_system_prompt()
        decisions = []
        confidences = []
        latencies = []

        for i, (_, row) in enumerate(papers.iterrows()):
            title = str(row.get("title", ""))
            abstract = str(row.get("abstract", ""))
            user = build_user_prompt(title, abstract)

            t0 = time.perf_counter()
            try:
                r = llm.query(system=system, user=user, schema=ScreeningResult)
                latency_ms = (time.perf_counter() - t0) * 1000
                decisions.append(r.relevant)
                confidences.append(r.confidence)
                latencies.append(latency_ms)
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                decisions.append(None)
                confidences.append(None)
                latencies.append(latency_ms)
                result["errors"] += 1

            if (i + 1) % 50 == 0:
                print(f"    Screening: {i+1}/{len(papers)}")

        result["screening"] = {
            "decisions": decisions,
            "confidences": confidences,
        }
        result["latencies_ms"] = latencies

    # Token usage
    result["total_input_tokens"] = llm.total_input_tokens
    result["total_output_tokens"] = llm.total_output_tokens
    result["total_calls"] = llm.total_calls

    usage = llm.get_usage_summary()
    result["estimated_cost_usd"] = usage.get("estimated_cost_usd", 0)

    return result


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def benchmark_providers(
    df: pd.DataFrame,
    n_sample: int = None,
    seed: int = 42,
    providers: list[str] = None,
    tasks: list[str] = None,
    gold_path: Path = None,
) -> dict:
    """
    Run the same papers through multiple providers and compare results.

    Args:
        df: Full DataFrame with papers
        n_sample: Number of papers to benchmark (default from config)
        seed: Random seed for sampling
        providers: List of provider names (None = auto-detect)
        tasks: List of tasks ["screen", "classify", "extract"]
        gold_path: Optional gold-standard CSV for accuracy metrics

    Returns:
        dict with comprehensive comparison results
    """
    if n_sample is None:
        n_sample = BENCHMARK_SAMPLE_SIZE
    if tasks is None:
        tasks = ["screen"]

    print(f"\n{'=' * 70}")
    print("CROSS-PROVIDER BENCHMARK")
    print(f"{'=' * 70}")

    # Detect providers
    if providers:
        available = [{"provider": p, "model": DEFAULT_MODELS.get(p, p)}
                     for p in providers]
    else:
        available = _detect_available_providers()

    if len(available) == 0:
        print("  ERROR: No providers detected. Set API keys or start Ollama.")
        return {}

    print(f"  Providers detected: {len(available)}")
    for p in available:
        print(f"    - {p['provider']}/{p['model']}")

    print(f"  Sample size: {n_sample}")
    print(f"  Tasks: {', '.join(tasks)}")

    # Sample papers
    sample = df.sample(min(n_sample, len(df)), random_state=seed)
    print(f"  Papers sampled: {len(sample)}")

    # Run each provider
    provider_results = {}
    for p_info in available:
        name = p_info["provider"]
        model = p_info["model"]
        key = f"{name}/{model}"

        try:
            result = _run_single_provider(name, model, sample, tasks)
            provider_results[key] = result
        except Exception as e:
            print(f"  ERROR running {key}: {e}")
            provider_results[key] = {"error": str(e)}

    # ── Compute comparison metrics ───────────────────────────────────
    comparison = {
        "n_papers": len(sample),
        "seed": seed,
        "tasks": tasks,
        "providers": {},
        "inter_provider_agreement": {},
    }

    # Per-provider summary
    for key, result in provider_results.items():
        if "error" in result:
            comparison["providers"][key] = {"error": result["error"]}
            continue

        latencies = result.get("latencies_ms", [])
        valid_latencies = [l for l in latencies if l is not None]

        provider_summary = {
            "provider": result["provider"],
            "model": result["model"],
            "total_calls": result.get("total_calls", 0),
            "total_input_tokens": result.get("total_input_tokens", 0),
            "total_output_tokens": result.get("total_output_tokens", 0),
            "estimated_cost_usd": result.get("estimated_cost_usd", 0),
            "errors": result.get("errors", 0),
        }

        if valid_latencies:
            provider_summary["median_latency_ms"] = round(np.median(valid_latencies), 1)
            provider_summary["p95_latency_ms"] = round(np.percentile(valid_latencies, 95), 1)
            provider_summary["mean_latency_ms"] = round(np.mean(valid_latencies), 1)

        # Cost per 1K papers (extrapolated)
        if result.get("estimated_cost_usd", 0) > 0 and len(sample) > 0:
            cost_per_paper = result["estimated_cost_usd"] / len(sample)
            provider_summary["cost_per_1k_papers_usd"] = round(cost_per_paper * 1000, 4)

        comparison["providers"][key] = provider_summary

    # Inter-provider agreement (screening only)
    if "screen" in tasks:
        screening_decisions = {}
        for key, result in provider_results.items():
            if "error" not in result and "screening" in result:
                decisions = result["screening"].get("decisions", [])
                # Filter out None decisions
                screening_decisions[key] = decisions

        # Pairwise kappa
        provider_keys = list(screening_decisions.keys())
        for a_key, b_key in combinations(provider_keys, 2):
            a_dec = screening_decisions[a_key]
            b_dec = screening_decisions[b_key]

            # Align on papers where both have valid decisions
            valid_mask = [(a is not None and b is not None)
                          for a, b in zip(a_dec, b_dec)]
            a_valid = np.array([d for d, v in zip(a_dec, valid_mask) if v])
            b_valid = np.array([d for d, v in zip(b_dec, valid_mask) if v])

            if len(a_valid) > 0:
                kappa = cohens_kappa(a_valid, b_valid)
                agree_pct = round(np.sum(a_valid == b_valid) / len(a_valid) * 100, 2)
                pair_key = f"{a_key}_vs_{b_key}"
                comparison["inter_provider_agreement"][pair_key] = {
                    "n": len(a_valid),
                    "kappa": kappa,
                    "interpretation": kappa_interpretation(kappa),
                    "agreement_pct": agree_pct,
                }

        # Accuracy against gold standard (if provided)
        if gold_path and gold_path.exists():
            gold = pd.read_csv(gold_path)
            gold_doi_set = set(gold["doi"].astype(str))
            sample_dois = sample["doi"].astype(str).tolist()

            # Find overlapping papers
            overlap_indices = [i for i, doi in enumerate(sample_dois) if doi in gold_doi_set]

            if len(overlap_indices) > 0:
                # Parse gold standard decisions
                gold_decisions = {}
                for _, row in gold.iterrows():
                    doi = str(row.get("doi", ""))
                    human_col = "adjudicated_relevant" if "adjudicated_relevant" in gold.columns \
                        else "human_relevant"
                    if pd.notna(row.get(human_col)):
                        human_val = str(row[human_col]).strip().lower()
                        gold_decisions[doi] = human_val in ("yes", "true", "1", "include")

                for key, result in provider_results.items():
                    if "error" in result or "screening" not in result:
                        continue

                    decisions = result["screening"].get("decisions", [])
                    tp = fp = fn = tn = 0
                    for idx in overlap_indices:
                        doi = sample_dois[idx]
                        if doi not in gold_decisions or idx >= len(decisions):
                            continue
                        if decisions[idx] is None:
                            continue

                        human = gold_decisions[doi]
                        pred = decisions[idx]

                        if human and pred:
                            tp += 1
                        elif not human and pred:
                            fp += 1
                        elif human and not pred:
                            fn += 1
                        else:
                            tn += 1

                    n_eval = tp + tn + fp + fn
                    if n_eval > 0:
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        acc = (tp + tn) / n_eval

                        comparison["providers"][key]["gold_standard"] = {
                            "n": n_eval,
                            "precision": round(prec, 4),
                            "recall": round(rec, 4),
                            "f1": round(f1, 4),
                            "accuracy": round(acc, 4),
                        }

    return comparison


def generate_provider_comparison_table(results: dict) -> pd.DataFrame:
    """Format benchmark results into a publication-ready comparison table."""
    rows = []
    for key, data in results.get("providers", {}).items():
        if "error" in data:
            continue
        row = {
            "Provider / Model": key,
            "Calls": data.get("total_calls", 0),
            "Input Tokens": data.get("total_input_tokens", 0),
            "Output Tokens": data.get("total_output_tokens", 0),
            "Cost (USD)": f"${data.get('estimated_cost_usd', 0):.4f}",
            "Cost/1K Papers": f"${data.get('cost_per_1k_papers_usd', 0):.4f}",
            "Median Latency (ms)": data.get("median_latency_ms", "—"),
            "P95 Latency (ms)": data.get("p95_latency_ms", "—"),
            "Errors": data.get("errors", 0),
        }

        # Gold standard metrics
        gold = data.get("gold_standard", {})
        if gold:
            row["F1 (vs Gold)"] = gold.get("f1", "—")
            row["Accuracy (vs Gold)"] = gold.get("accuracy", "—")
        else:
            row["F1 (vs Gold)"] = "—"
            row["Accuracy (vs Gold)"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-provider LLM benchmark")
    parser.add_argument("--n", type=int, default=BENCHMARK_SAMPLE_SIZE,
                        help=f"Number of papers to benchmark (default: {BENCHMARK_SAMPLE_SIZE})")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated list of providers (default: auto-detect)")
    parser.add_argument("--tasks", type=str, default="screen",
                        help="Comma-separated tasks: screen,classify,extract (default: screen)")
    parser.add_argument("--gold", type=Path, default=None,
                        help="Gold standard CSV for accuracy metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print(f"  Run the main pipeline first to generate filtered results.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} papers from {INPUT_FILE}")

    providers = args.providers.split(",") if args.providers else None
    tasks = args.tasks.split(",")

    results = benchmark_providers(
        df=df,
        n_sample=args.n,
        seed=args.seed,
        providers=providers,
        tasks=tasks,
        gold_path=args.gold,
    )

    if not results:
        sys.exit(1)

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save comparison table CSV
    table = generate_provider_comparison_table(results)
    if len(table) > 0:
        table.to_csv(OUTPUT_CSV, index=False)

    # Print inter-provider agreement
    agreements = results.get("inter_provider_agreement", {})
    if agreements:
        print(f"\n{'─' * 70}")
        print("INTER-PROVIDER AGREEMENT")
        print(f"{'─' * 70}")
        for pair, metrics in agreements.items():
            print(f"  {pair}: k={metrics['kappa']:.3f} ({metrics['interpretation']}), "
                  f"agree={metrics['agreement_pct']:.1f}%")

    print(f"\n  Results saved to: {OUTPUT_JSON}")
    if len(table) > 0:
        print(f"  Comparison table: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
