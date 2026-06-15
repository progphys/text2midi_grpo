#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot critic score distributions for base vs finetuned runs.")
    parser.add_argument("--results-json", required=True, help="Path to judge_eval results.json")
    parser.add_argument(
        "--output-image",
        default="outputs/plots/critic_distributions.png",
        help="Where to save the plot image.",
    )
    parser.add_argument(
        "--output-summary",
        default="outputs/plots/critic_distribution_summary.json",
        help="Where to save quantile summary JSON.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metric(records: list[dict], model_name: str, metric_name: str) -> np.ndarray:
    values: list[float] = []
    for item in records:
        if item["model_name"] != model_name:
            continue
        value = item["critics"].get(metric_name)
        if value is not None:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def paired_deltas(records: list[dict], metric_name: str) -> np.ndarray:
    by_prompt: dict[int, dict[str, float]] = {}
    for item in records:
        by_prompt.setdefault(item["prompt_index"], {})[item["model_name"]] = float(item["critics"][metric_name])
    deltas: list[float] = []
    for prompt_index in sorted(by_prompt):
        group = by_prompt[prompt_index]
        if "base" in group and "final" in group:
            deltas.append(group["final"] - group["base"])
    return np.asarray(deltas, dtype=float)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_path = resolve_path(project_root, args.results_json)
    output_image = resolve_path(project_root, args.output_image)
    output_summary = resolve_path(project_root, args.output_summary)

    records = load_records(results_path)

    metrics = {
        "observer_fixed_pairs_raw": {
            "title": "observer_fixed_pairs_raw",
            "base": extract_metric(records, "base", "observer_fixed_pairs_raw"),
            "final": extract_metric(records, "final", "observer_fixed_pairs_raw"),
        },
        "final_observer_raw": {
            "title": "final_observer_raw",
            "base": extract_metric(records, "base", "final_observer_raw"),
            "final": extract_metric(records, "final", "final_observer_raw"),
        },
    }

    summary_payload = {
        metric_name: {
            "base": summarize(payload["base"]),
            "final": summarize(payload["final"]),
            "delta_final_minus_base": summarize(paired_deltas(records, metric_name)),
        }
        for metric_name, payload in metrics.items()
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"base": "#1f77b4", "final": "#d62728"}

    for ax, (metric_name, payload) in zip(axes[0], metrics.items()):
        all_values = np.concatenate([payload["base"], payload["final"]])
        bins = np.linspace(np.min(all_values), np.max(all_values), 14)
        ax.hist(payload["base"], bins=bins, alpha=0.55, color=colors["base"], label="base", density=True)
        ax.hist(payload["final"], bins=bins, alpha=0.55, color=colors["final"], label="finetuned", density=True)
        ax.axvline(np.mean(payload["base"]), color=colors["base"], linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(payload["final"]), color=colors["final"], linestyle="--", linewidth=1.5)
        ax.set_title(f"Distribution: {payload['title']}")
        ax.set_xlabel("Critic score")
        ax.set_ylabel("Density")
        ax.legend()

    # ECDFs make the shift between distributions easier to compare.
    for metric_name, payload in metrics.items():
        x_base, y_base = ecdf(payload["base"])
        x_final, y_final = ecdf(payload["final"])
        axes[1, 0].plot(x_base, y_base, label=f"{metric_name}: base", linewidth=2)
        axes[1, 0].plot(x_final, y_final, linestyle="--", label=f"{metric_name}: finetuned", linewidth=2)
    axes[1, 0].set_title("ECDF Comparison")
    axes[1, 0].set_xlabel("Critic score")
    axes[1, 0].set_ylabel("Cumulative fraction")
    axes[1, 0].legend(fontsize=8)

    delta_observer = paired_deltas(records, "observer_fixed_pairs_raw")
    delta_final = paired_deltas(records, "final_observer_raw")
    x = np.arange(delta_observer.size)
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].scatter(x, delta_observer, label="observer_fixed_pairs_raw", color="#2ca02c", s=28, alpha=0.85)
    axes[1, 1].scatter(x, delta_final, label="final_observer_raw", color="#9467bd", s=28, alpha=0.85)
    axes[1, 1].set_title("Prompt-wise Delta (finetuned - base)")
    axes[1, 1].set_xlabel("Prompt index")
    axes[1, 1].set_ylabel("Delta score")
    axes[1, 1].legend()

    fig.suptitle("Critic Distributions: Base vs Finetuned", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "results_json": str(results_path),
        "output_image": str(output_image),
        "output_summary": str(output_summary),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
