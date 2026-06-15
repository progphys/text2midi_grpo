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
    parser = argparse.ArgumentParser(description="Evaluate Text2MIDI checkpoints")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--experiment", choices=available_experiments())
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from text2midi.eval_app import run_evaluation

    run_evaluation(args)
