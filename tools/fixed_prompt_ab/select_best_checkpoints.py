#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ab_test_final_observer.common import dump_json, resolve_path


DEFAULT_EXPERIMENTS = [
    ("rules_fixed", "ffn_key_moderate_density_drumratio_nomenter_fixedprompt_len800_debug"),
    ("rules_curriculum", "ffn_curriculum_4stage_interleaved_len800_r18"),
    ("critic_only_final_final", "final_final_only_ffn_fixedprompt_25rollouts_len800"),
    ("critic_after_curriculum", "final_final_from_curriculum_ffn_fixedprompt_len800_r18"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best fixed-prompt checkpoints from saved rollout manifests."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment to include. Use either EXPERIMENT or LABEL=EXPERIMENT. Defaults to the main four.",
    )
    parser.add_argument("--output-json", default="outputs/model_registry/fixed_prompt_best_checkpoints.json")
    parser.add_argument(
        "--checkpoint-stat",
        choices=["top1", "mean"],
        default="top1",
        help="Statistic used to select the checkpoint. top1 means best rollout on the step.",
    )
    return parser.parse_args()


def parse_experiments(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_EXPERIMENTS
    parsed = []
    for value in values:
        if "=" in value:
            label, experiment = value.split("=", 1)
        else:
            experiment = value
            label = re.sub(r"[^a-zA-Z0-9_]+", "_", experiment).strip("_")
        parsed.append((label, experiment))
    return parsed


def load_manifest(experiment: str) -> list[dict]:
    path = PROJECT_ROOT / "outputs" / "text2midi" / experiment / "midis_manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Rollout manifest not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def raw_reward_keys(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        reward = row.get("reward") or {}
        for key, value in reward.items():
            if key.endswith("_raw") and key != "observer_raw" and isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def checkpoint_exists(experiment: str, step: int) -> bool:
    return (PROJECT_ROOT / "outputs" / "checkpoints" / experiment / f"step_{step:05d}").is_dir()


def project_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def aggregate_step(rows: list[dict], score_key: str) -> dict[int, dict]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        reward = row.get("reward") or {}
        value = reward.get(score_key)
        if isinstance(value, (int, float)):
            by_step[int(row["step"])].append(float(value))

    out = {}
    for step, values in sorted(by_step.items()):
        if not values:
            continue
        out[step] = {
            "count": len(values),
            "mean": mean(values),
            "top1": max(values),
        }
    return out


def select_for_experiment(label: str, experiment: str, checkpoint_stat: str) -> dict:
    rows = load_manifest(experiment)
    raw_keys = raw_reward_keys(rows)
    score_key = raw_keys[0] if raw_keys else "total"
    metric_name = f"reward.{score_key}"
    step_stats = aggregate_step(rows, score_key)
    checkpoint_steps = [step for step in step_stats if checkpoint_exists(experiment, step)]
    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoint step found for experiment={experiment}")

    selected_step = max(checkpoint_steps, key=lambda step: step_stats[step][checkpoint_stat])
    selected_checkpoint = PROJECT_ROOT / "outputs" / "checkpoints" / experiment / f"step_{selected_step:05d}"

    return {
        "label": label,
        "experiment": experiment,
        "score_key": score_key,
        "selection_metric": f"{metric_name}.{checkpoint_stat}",
        "selected_step": selected_step,
        "selected_value": step_stats[selected_step][checkpoint_stat],
        "selected_checkpoint": project_relative(selected_checkpoint),
        "checkpoint_stat": checkpoint_stat,
        "available_checkpoint_steps": [
            {
                "step": step,
                "checkpoint": project_relative(PROJECT_ROOT / "outputs" / "checkpoints" / experiment / f"step_{step:05d}"),
                **step_stats[step],
            }
            for step in checkpoint_steps
        ],
    }


def main() -> None:
    args = parse_args()
    experiments = parse_experiments(args.experiment)
    selections = [select_for_experiment(label, experiment, args.checkpoint_stat) for label, experiment in experiments]
    payload = {
        "checkpoint_stat": args.checkpoint_stat,
        "selection_note": "Rule-based runs use top1 reward.total; critic-only runs use top1 raw critic score.",
        "models": selections,
    }
    output_path = resolve_path(args.output_json)
    assert output_path is not None
    dump_json(output_path, payload)
    print(str(output_path.resolve()))
    for item in selections:
        print(
            f"{item['label']}: step={item['selected_step']:05d} "
            f"{item['selection_metric']}={item['selected_value']:.6f}"
        )


if __name__ == "__main__":
    main()
