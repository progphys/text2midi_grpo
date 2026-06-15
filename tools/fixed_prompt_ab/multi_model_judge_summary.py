#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ab_test_final_observer.common import dump_json, load_json, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-model fixed-prompt LLM judge results.")
    parser.add_argument("--results-json", required=True, help="results.json from tools/grok_judge/grok_midi_judge.py")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def summarize_scores(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "mean": None, "median": None, "top1": None, "min": None, "max": None}
    return {
        "count": len(scores),
        "mean": mean(scores),
        "median": median(scores),
        "top1": max(scores),
        "min": min(scores),
        "max": max(scores),
    }


def pairwise(a_scores: list[float], b_scores: list[float]) -> dict:
    wins = 0.0
    total = 0
    for a in a_scores:
        for b in b_scores:
            total += 1
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    dominance = wins / total if total else None
    return {
        "pair_count": total,
        "a_pairwise_dominance": dominance,
        "delta_mean": (mean(a_scores) - mean(b_scores)) if a_scores and b_scores else None,
        "delta_top1": (max(a_scores) - max(b_scores)) if a_scores and b_scores else None,
    }


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_json)
    assert results_path is not None
    output_path = resolve_path(args.output_json) if args.output_json else results_path.parent / "multi_model_ab_summary.json"
    assert output_path is not None

    payload = load_json(results_path)
    rows = payload.get("results", [])
    by_group: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        group = row.get("group")
        if not group:
            continue
        if row.get("error") or row.get("score") is None:
            errors[str(group)] += 1
            continue
        by_group[str(group)].append(float(row["score"]))

    group_summary = {
        group: {
            **summarize_scores(scores),
            "errors": errors.get(group, 0),
        }
        for group, scores in sorted(by_group.items())
    }

    comparisons = {}
    for a, b in itertools.permutations(sorted(by_group), 2):
        comparisons[f"{a}__vs__{b}"] = {
            "a": a,
            "b": b,
            **pairwise(by_group[a], by_group[b]),
        }

    base_vs_models = {}
    if "base" in by_group:
        for model in sorted(group for group in by_group if group != "base"):
            model_vs_base = pairwise(by_group[model], by_group["base"])
            base_vs_model = pairwise(by_group["base"], by_group[model])
            base_vs_models[model] = {
                "model": model,
                "base_count": len(by_group["base"]),
                "model_count": len(by_group[model]),
                "model_mean": mean(by_group[model]) if by_group[model] else None,
                "base_mean": mean(by_group["base"]) if by_group["base"] else None,
                "model_top1": max(by_group[model]) if by_group[model] else None,
                "base_top1": max(by_group["base"]) if by_group["base"] else None,
                "model_pairwise_dominance_over_base": model_vs_base["a_pairwise_dominance"],
                "base_pairwise_dominance_over_model": base_vs_model["a_pairwise_dominance"],
                "delta_mean_model_minus_base": model_vs_base["delta_mean"],
                "delta_top1_model_minus_base": model_vs_base["delta_top1"],
                "pair_count": model_vs_base["pair_count"],
            }

    ranking_by_top1 = sorted(
        group_summary,
        key=lambda group: (group_summary[group]["top1"] is not None, group_summary[group]["top1"]),
        reverse=True,
    )
    ranking_by_mean = sorted(
        group_summary,
        key=lambda group: (group_summary[group]["mean"] is not None, group_summary[group]["mean"]),
        reverse=True,
    )

    summary = {
        "input_results_json": str(results_path.resolve()),
        "group_summary": group_summary,
        "base_vs_models": base_vs_models,
        "pairwise_comparisons": comparisons,
        "ranking_by_top1": ranking_by_top1,
        "ranking_by_mean": ranking_by_mean,
    }
    dump_json(output_path, summary)
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
