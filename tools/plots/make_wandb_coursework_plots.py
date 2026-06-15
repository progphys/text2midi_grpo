#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENTITY = "udalov0078-"
PROJECT = "text2midi-grpo"

RUNS = {
    "rules_fixed": {
        "id": "6wep1e5x",
        "label": "Правила, один промпт",
        "family": "fixed",
    },
    "curriculum": {
        "id": "nl8u8m4g",
        "label": "Поэтапное обучение",
        "family": "fixed",
    },
    "critic_only": {
        "id": "i17jcg5n",
        "label": "Критик от базы",
        "family": "critic",
    },
    "critic_after_curriculum": {
        "id": "058m3xcn",
        "label": "Критик после curriculum",
        "family": "critic",
    },
    "critic_v2_low_lr": {
        "id": "nn9trrnr",
        "label": "Критик v2, малый шаг",
        "family": "critic",
    },
    "track_select_len800": {
        "id": "a9lzd3as",
        "label": "800 токенов",
        "family": "length",
    },
    "track_select_len1200": {
        "id": "julfyiv3",
        "label": "1200 токенов",
        "family": "length",
    },
    "track_select_len1500": {
        "id": "8kseyjly",
        "label": "1500 токенов",
        "family": "length",
    },
    "moderate_meter": {
        "id": "n6rep5la",
        "label": "С шаблоном размера",
        "family": "narrow",
    },
    "moderate_no_meter": {
        "id": "icpa5uaw",
        "label": "Без шаблона размера",
        "family": "narrow",
    },
    "broad_critic_ffn": {
        "id": "fch90yhi",
        "label": "Разнообразные prompt, FFN",
        "family": "broad",
    },
    "broad_rules_critic": {
        "id": "mon3q60b",
        "label": "Разнообразные prompt, правила + критик",
        "family": "broad",
    },
}

METRIC_LABELS = {
    "reward/total": "Общая награда",
    "reward/key_profile": "Тональность",
    "reward/meter_template": "Размер",
    "reward/moderate_note_density": "Умеренная плотность нот",
    "reward/note_density": "Плотность нот",
    "reward/drum_note_ratio": "Доля ударных нот",
    "reward/duration_balance": "Длительность",
    "metric/final_final_raw": "Сырая оценка критика",
    "metric/final_final_top1_raw": "Лучшая сырая оценка критика",
    "metric/final_observer_raw": "Сырая оценка критика v2",
    "metric/final_observer_top1_raw": "Лучшая сырая оценка критика v2",
    "loss": "Функция потерь",
    "kl": "KL-расхождение",
    "grad_norm": "Норма градиента",
    "train/lr": "Шаг обучения",
    "metrics/valid_rate": "Доля валидных MIDI",
}


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def fetch_history(api: wandb.Api, run_id: str, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    cache_path = cache_dir / f"{run_id}.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = list(run.scan_history(page_size=500))
    if not rows:
        raise RuntimeError(f"No history rows for run {run_id}")
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def prepare_histories(cache_dir: Path, refresh: bool) -> dict[str, pd.DataFrame]:
    load_env()
    api = wandb.Api()
    histories: dict[str, pd.DataFrame] = {}
    for key, meta in RUNS.items():
        try:
            histories[key] = fetch_history(api, meta["id"], cache_dir, refresh)
        except Exception as exc:  # noqa: BLE001 - keep plotting useful if one run is inaccessible.
            print(f"[warn] skip {key} ({meta['id']}): {exc}")
    return histories


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.8),
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_lines(
    histories: dict[str, pd.DataFrame],
    run_keys: list[str],
    metric: str,
    title: str,
    output_path: Path,
    *,
    ylabel: str | None = None,
    smooth_window: int = 1,
    stage_lines: bool = False,
) -> None:
    fig, ax = plt.subplots()
    plotted = False
    for key in run_keys:
        df = histories.get(key)
        if df is None or metric not in df.columns or "_step" not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        steps = pd.to_numeric(df["_step"], errors="coerce")
        mask = steps.notna() & series.notna()
        if not mask.any():
            continue
        ax.plot(
            steps[mask],
            smooth(series[mask], smooth_window),
            marker="o",
            linewidth=2,
            markersize=3,
            label=RUNS[key]["label"],
        )
        plotted = True

    if stage_lines:
        for step, label in [(10, "тональность"), (20, "размер"), (30, "стиль")]:
            ax.axvline(step, color="#555555", linewidth=1, alpha=0.35)
            ax.text(step + 0.3, ax.get_ylim()[1], label, va="top", fontsize=8, color="#444444")

    ax.set_title(title)
    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel(ylabel or METRIC_LABELS.get(metric, metric))
    if plotted:
        ax.legend(frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_rule_components(histories: dict[str, pd.DataFrame], key: str, title: str, output_path: Path) -> None:
    metrics = [
        "reward/total",
        "reward/key_profile",
        "reward/moderate_note_density",
        "reward/drum_note_ratio",
        "reward/duration_balance",
    ]
    fig, ax = plt.subplots()
    df = histories.get(key)
    if df is not None and "_step" in df.columns:
        steps = pd.to_numeric(df["_step"], errors="coerce")
        for metric in metrics:
            if metric not in df.columns:
                continue
            values = pd.to_numeric(df[metric], errors="coerce")
            mask = steps.notna() & values.notna()
            if mask.any():
                ax.plot(steps[mask], values[mask], marker="o", linewidth=2, markersize=3, label=METRIC_LABELS[metric])
    ax.set_title(title)
    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel("Значение компоненты")
    ax.legend(frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_training_stability(histories: dict[str, pd.DataFrame], run_keys: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    metrics = ["loss", "kl", "grad_norm", "train/lr"]
    for ax, metric in zip(axes.ravel(), metrics):
        for key in run_keys:
            df = histories.get(key)
            if df is None or metric not in df.columns or "_step" not in df.columns:
                continue
            steps = pd.to_numeric(df["_step"], errors="coerce")
            values = pd.to_numeric(df[metric], errors="coerce")
            mask = steps.notna() & values.notna()
            if mask.any():
                ax.plot(steps[mask], values[mask], linewidth=1.8, label=RUNS[key]["label"])
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Шаг")
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Стабильность GRPO-обучения", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Russian matplotlib plots from selected W&B runs.")
    parser.add_argument("--out-dir", default="paper/figures/wandb")
    parser.add_argument("--cache-dir", default="outputs/wandb_history_cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    cache_dir = PROJECT_ROOT / args.cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    histories = prepare_histories(cache_dir, args.refresh)

    plot_rule_components(
        histories,
        "rules_fixed",
        "Обучение по правилам на одном промпте",
        out_dir / "01_fixed_prompt_rule_components",
    )
    plot_lines(
        histories,
        ["curriculum"],
        "reward/total",
        "Поэтапное обучение: динамика общей награды",
        out_dir / "02_curriculum_reward_total",
        stage_lines=True,
    )
    plot_rule_components(
        histories,
        "curriculum",
        "Поэтапное обучение: компоненты награды",
        out_dir / "03_curriculum_reward_components",
    )
    plot_lines(
        histories,
        ["critic_only", "critic_after_curriculum"],
        "metric/final_final_top1_raw",
        "Обучение по критику: лучшая сырая оценка",
        out_dir / "04_critic_top1_raw",
        ylabel="Лучшая сырая оценка критика",
    )
    plot_lines(
        histories,
        ["critic_only", "critic_after_curriculum"],
        "metric/final_final_raw",
        "Обучение по критику: средняя сырая оценка",
        out_dir / "05_critic_mean_raw",
        ylabel="Средняя сырая оценка критика",
    )
    plot_lines(
        histories,
        ["moderate_meter", "moderate_no_meter"],
        "reward/total",
        "Узкие prompt: влияние награды за метрический шаблон",
        out_dir / "06_meter_vs_no_meter_total",
    )
    plot_lines(
        histories,
        ["track_select_len800", "track_select_len1200", "track_select_len1500"],
        "reward/total",
        "Длина генерации: 800, 1200 и 1500 токенов",
        out_dir / "07_generation_length_total_reward",
    )
    plot_lines(
        histories,
        ["broad_critic_ffn", "broad_rules_critic"],
        "reward/total",
        "Разнообразные prompt: шумность общей награды",
        out_dir / "08_broad_prompt_reward_total",
    )
    plot_training_stability(
        histories,
        ["rules_fixed", "curriculum", "critic_only", "critic_after_curriculum", "critic_v2_low_lr"],
        out_dir / "09_training_stability",
    )
    plot_lines(
        histories,
        ["critic_v2_low_lr"],
        "metric/final_observer_top1_raw",
        "Критик v2: лучшая сырая оценка при малом шаге обучения",
        out_dir / "10_critic_v2_low_lr_top1_raw",
        ylabel="Лучшая сырая оценка критика v2",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
