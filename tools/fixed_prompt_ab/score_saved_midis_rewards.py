#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from symusic import Score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config import train_config
from core.critics.metrics import build_default_metric_critics, score_symbolic_scores_with_critics
from core.rewards import batch_rewards
from tools.ab_test_final_observer.common import dump_json, load_json, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved fixed-prompt MIDI files with internal rewards/critics.")
    parser.add_argument("--generations-json", default="outputs/fixed_prompt_ab/fixed_prompt_top1_len800_90x/generations.json")
    parser.add_argument(
        "--reward-experiment",
        default="ffn_key_moderate_density_drumratio_nomenter_fixedprompt_len800_debug",
        help="Experiment whose reward config is used for rule-based scoring.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--critics",
        default="",
        help="Comma-separated critic metric names from build_default_metric_critics, e.g. final_observer,final_final.",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "top1": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "top1": max(values),
        "min": min(values),
        "max": max(values),
    }


def pairwise(a_values: list[float], b_values: list[float]) -> dict:
    wins = 0.0
    total = 0
    for a, b in itertools.product(a_values, b_values):
        total += 1
        if a > b:
            wins += 1.0
        elif a == b:
            wins += 0.5
    return {
        "pair_count": total,
        "a_pairwise_dominance": wins / total if total else None,
        "delta_mean": mean(a_values) - mean(b_values) if a_values and b_values else None,
        "delta_top1": max(a_values) - max(b_values) if a_values and b_values else None,
    }


def grouped_top1(values: list[float], group_size: int) -> list[float]:
    if group_size <= 1:
        return list(values)
    return [
        max(values[start : start + group_size])
        for start in range(0, len(values), group_size)
        if values[start : start + group_size]
    ]


def build_summary(rows: list[dict], metric: str) -> dict:
    by_group: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        group = str(row.get("model_name") or row.get("model_label") or "")
        if not group:
            continue
        value = row.get(metric)
        if value is None:
            errors[group] += 1
            continue
        by_group[group].append(float(value))

    group_summary = {
        group: {
            **summarize(values),
            "errors": errors.get(group, 0),
        }
        for group, values in sorted(by_group.items())
    }

    base_vs_models = {}
    if "base" in by_group:
        for group in sorted(name for name in by_group if name != "base"):
            model_vs_base = pairwise(by_group[group], by_group["base"])
            model_top1_groups = grouped_top1(by_group[group], group_size=30)
            base_top1_groups = grouped_top1(by_group["base"], group_size=30)
            model_top1_vs_base_top1 = pairwise(model_top1_groups, base_top1_groups)
            base_vs_models[group] = {
                "model": group,
                "metric": metric,
                "model_count": len(by_group[group]),
                "base_count": len(by_group["base"]),
                "model_mean": mean(by_group[group]) if by_group[group] else None,
                "base_mean": mean(by_group["base"]) if by_group["base"] else None,
                "model_top1": max(by_group[group]) if by_group[group] else None,
                "base_top1": max(by_group["base"]) if by_group["base"] else None,
                "model_pairwise_dominance_over_base": model_vs_base["a_pairwise_dominance"],
                "delta_mean_model_minus_base": model_vs_base["delta_mean"],
                "delta_top1_model_minus_base": model_vs_base["delta_top1"],
                "pair_count": model_vs_base["pair_count"],
                "top1_group_size": 30,
                "model_top1_group_count": len(model_top1_groups),
                "base_top1_group_count": len(base_top1_groups),
                "model_top1_group_values": model_top1_groups,
                "base_top1_group_values": base_top1_groups,
                "model_top1_mean": mean(model_top1_groups) if model_top1_groups else None,
                "base_top1_mean": mean(base_top1_groups) if base_top1_groups else None,
                "model_top1_pairwise_dominance_over_base_top1": model_top1_vs_base_top1["a_pairwise_dominance"],
                "delta_top1_mean_model_minus_base": model_top1_vs_base_top1["delta_mean"],
            }

    ranking_by_mean = sorted(
        group_summary,
        key=lambda group: (group_summary[group]["mean"] is not None, group_summary[group]["mean"]),
        reverse=True,
    )
    ranking_by_top1 = sorted(
        group_summary,
        key=lambda group: (group_summary[group]["top1"] is not None, group_summary[group]["top1"]),
        reverse=True,
    )

    return {
        "metric": metric,
        "group_summary": group_summary,
        "base_vs_models": base_vs_models,
        "ranking_by_mean": ranking_by_mean,
        "ranking_by_top1": ranking_by_top1,
    }


def main() -> None:
    args = parse_args()
    generations_path = resolve_path(args.generations_json)
    assert generations_path is not None
    output_path = (
        resolve_path(args.output_json)
        if args.output_json
        else generations_path.parent / f"reward_scores_{args.reward_experiment}.json"
    )
    assert output_path is not None

    rows = load_json(generations_path)
    cfg = train_config("text2midi", args.reward_experiment)

    scores = []
    captions = []
    valid_indices = []
    for idx, row in enumerate(rows):
        row["reward_score_error"] = None
        if not row.get("decode_ok") or not row.get("midi_path"):
            row["reward_score_error"] = "decode_failed"
            continue
        midi_path = Path(str(row["midi_path"]))
        if not midi_path.exists():
            row["reward_score_error"] = f"midi_missing:{midi_path}"
            continue
        try:
            scores.append(Score(str(midi_path)))
            captions.append(str(row.get("prompt") or ""))
            valid_indices.append(idx)
        except Exception as exc:  # noqa: BLE001
            row["reward_score_error"] = f"score_load_failed:{exc}"

    reward_dicts = batch_rewards(scores, cfg.reward, captions=captions) if scores else []
    for idx, reward in zip(valid_indices, reward_dicts):
        rows[idx]["reward_experiment"] = args.reward_experiment
        rows[idx]["reward"] = reward
        for key, value in reward.items():
            rows[idx][f"reward/{key}"] = value

    requested_critics = [item.strip() for item in args.critics.split(",") if item.strip()]
    if requested_critics and scores:
        specs = [spec for spec in build_default_metric_critics(PROJECT_ROOT) if spec.name in set(requested_critics)]
        missing = sorted(set(requested_critics) - {spec.name for spec in specs})
        if missing:
            raise ValueError(f"Unknown critic metric(s): {missing}")
        critic_rows = score_symbolic_scores_with_critics(
            PROJECT_ROOT,
            specs,
            scores=scores,
            captions=captions,
            group_size=None,
            tmp_prefix="fixed_prompt_reward_eval_",
        )
        for idx, critic_values in zip(valid_indices, critic_rows):
            rows[idx]["critic_metrics"] = critic_values
            for key, value in critic_values.items():
                rows[idx][f"critic/{key}"] = value

    metrics = ["reward/total"]
    if requested_critics:
        for critic in requested_critics:
            metrics.extend([f"critic/{critic}", f"critic/{critic}_raw"])

    summary = {
        "input_generations_json": str(generations_path.resolve()),
        "reward_experiment": args.reward_experiment,
        "critics": requested_critics,
        "metrics": {metric: build_summary(rows, metric) for metric in metrics},
    }
    dump_json(output_path, {"rows": rows, "summary": summary})
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()
