#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def available_experiments() -> list[str]:
    exp_dir = PROJECT_ROOT / "configs" / "text2midi" / "train" / "experiments"
    return sorted(path.stem for path in exp_dir.glob("*.yaml"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO experiments for Text2MIDI")
    parser.add_argument("--experiment", required=True, choices=available_experiments())
    parser.add_argument("--num-rollouts", type=int)
    parser.add_argument("--rollout-max-len", type=int)
    parser.add_argument("--generation-chunk-size", type=int)
    parser.add_argument("--update-mini-batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf-style override, e.g. --set training.lr=1e-4",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from text2midi.train_app import run_training

    run_training(args)
