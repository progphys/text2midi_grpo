#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ab_test_final_observer.common import (
    bootstrap_mean_ci,
    dump_json,
    exact_sign_test_pvalue,
    load_json,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate LLM judge results into prompt-level A/B statistics.")
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--output-prompt-level", default=None)
    return parser.parse_args()


def _prompt_index_from_pair_key(pair_key: str) -> int | None:
    match = re.search(r"prompt_(\d+)", str(pair_key))
    if not match:
        return None
    return int(match.group(1))


def _load_prompt_text(row: dict) -> str | None:
    midi_path = row.get("midi_path")
    if not midi_path:
        return None
    prompt_path = Path(str(midi_path)).resolve().parent / "prompt.txt"
    if not prompt_path.exists():
        return None
    return prompt_path.read_text(encoding="utf-8").strip()


def main():
    args = parse_args()
    results_path = resolve_path(args.results_json)
    output_summary = resolve_path(args.output_summary) if args.output_summary else results_path.parent / "ab_summary.json"
    output_prompt_level = (
        resolve_path(args.output_prompt_level) if args.output_prompt_level else results_path.parent / "ab_prompt_level.json"
    )

    payload = load_json(results_path)
    rows = payload.get("results", [])

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"base": [], "final": []})
    for row in rows:
        pair_key = row.get("pair_key")
        group = row.get("group")
        if not pair_key or group not in {"base", "final"}:
            continue
        grouped[str(pair_key)][str(group)].append(row)

    prompt_level: list[dict] = []
    mean_deltas: list[float] = []
    best_deltas: list[float] = []
    verdicts: list[str] = []

    for pair_key in sorted(grouped):
        base_rows = [r for r in grouped[pair_key]["base"] if r.get("score") is not None and not r.get("error")]
        final_rows = [r for r in grouped[pair_key]["final"] if r.get("score") is not None and not r.get("error")]
        base_scores = [float(r["score"]) for r in base_rows]
        final_scores = [float(r["score"]) for r in final_rows]
        if not base_scores or not final_scores:
            continue

        base_mean = mean(base_scores)
        final_mean = mean(final_scores)
        base_best = max(base_scores)
        final_best = max(final_scores)

        wins = 0.0
        total = 0
        for f in final_scores:
            for b in base_scores:
                total += 1
                if f > b:
                    wins += 1.0
                elif f == b:
                    wins += 0.5
        dominance = wins / total
        if dominance > 0.5:
            verdict = "final"
        elif dominance < 0.5:
            verdict = "base"
        else:
            verdict = "tie"

        prompt_level.append(
            {
                "pair_key": pair_key,
                "prompt_index": _prompt_index_from_pair_key(pair_key),
                "prompt": _load_prompt_text(base_rows[0]) or _load_prompt_text(final_rows[0]),
                "base_scores": base_scores,
                "final_scores": final_scores,
                "base_mean": base_mean,
                "final_mean": final_mean,
                "delta_mean": final_mean - base_mean,
                "base_best": base_best,
                "final_best": final_best,
                "delta_best": final_best - base_best,
                "pairwise_dominance_final_over_base": dominance,
                "verdict": verdict,
            }
        )
        mean_deltas.append(final_mean - base_mean)
        best_deltas.append(final_best - base_best)
        verdicts.append(verdict)

    wins = verdicts.count("final")
    losses = verdicts.count("base")
    ties = verdicts.count("tie")
    total_prompts = len(prompt_level)
    summary = {
        "prompt_count": total_prompts,
        "final_win_count": wins,
        "base_win_count": losses,
        "tie_count": ties,
        "final_win_rate": wins / total_prompts if total_prompts else 0.0,
        "base_win_rate": losses / total_prompts if total_prompts else 0.0,
        "tie_rate": ties / total_prompts if total_prompts else 0.0,
        "sign_test_pvalue_no_ties": exact_sign_test_pvalue(wins, losses),
        "delta_mean_of_3": {
            "median": median(mean_deltas) if mean_deltas else None,
            **bootstrap_mean_ci(mean_deltas),
        },
        "delta_best_of_3": {
            "median": median(best_deltas) if best_deltas else None,
            **bootstrap_mean_ci(best_deltas),
        },
    }

    dump_json(output_prompt_level, prompt_level)
    dump_json(output_summary, summary)
    print(str(output_summary.resolve()))


if __name__ == "__main__":
    main()
